import { CONFIG } from "./config.mjs";
import { log } from "./log.mjs";
import { tf } from "./tf.mjs";


function clipGradients(grads, clipValue) {
    const clipped = {};
    const toDispose = [];
    for (const key in grads) {
        if (grads[key] !== null) {
            clipped[key] = tf.clipByValue(grads[key], -clipValue, clipValue);
            toDispose.push(grads[key]);
        }
    }
    tf.dispose(toDispose);
    return clipped;
}
function sampleAction(mu, logStd) {
    const logStdClamped = tf.clipByValue(logStd, -20, 2);
    const std = tf.exp(logStdClamped);

    // Reparameterization trick: u = mu + std * epsilon
    const epsilon = tf.randomNormal(mu.shape);
    const u = mu.add(std.mul(epsilon));
    const action = tf.tanh(u);

    // Calculate Log Probability
    // 1. Gaussian Log Prob of 'u'
    // Formula: -0.5 * (((x - mu) / std)^2 + 2*log(std) + log(2*PI))
    const variance = tf.square(std);
    const logScale = logStdClamped;
    const logPi = tf.scalar(Math.log(2 * Math.PI));

    const logProbGaussian = tf.scalar(-0.5).mul(
        tf.square(u.sub(mu)).div(variance)
            .add(tf.scalar(2).mul(logScale))
            .add(logPi)
    );

    // 2. Tanh Correction (Jacobian adjustment)
    // Formula: log_prob(a) = log_prob(u) - sum(log(1 - tanh(u)^2))
    const logProbCorrection = tf.log(tf.scalar(1).sub(tf.square(action)).add(1e-6));

    const logProb = logProbGaussian.sub(logProbCorrection).sum(1, true);

    return { action, logProb };
}

export async function train(models, batch, importanceWeights = null) {
    if (batch.length < CONFIG.BATCH_SIZE) return null;

    const { actor, critic1, critic2, targetCritic1, targetCritic2, optimizerActor, optimizerCritic1, optimizerCritic2, logAlpha, optimizerAlpha } = models;
    const gamma = CONFIG.DISCOUNT_FACTOR;
    const rewardScale = CONFIG.REWARD_SCALE || 1.0;
    const gradClip = CONFIG.GRADIENT_CLIP || 0.5;
    const actorDelay = CONFIG.ACTOR_UPDATE_DELAY || 2;
    const targetEntropy = CONFIG.TARGET_ENTROPY || -CONFIG.ACTION_SIZE;

    const alpha = logAlpha.exp();
    const states = tf.tensor2d(batch.map(b => b.state));
    const nextStates = tf.tensor2d(batch.map(b => b.nextState));
    const actions = tf.tensor2d(batch.map(b => b.action));
    const rewards = tf.tensor1d(batch.map(b => b.reward * rewardScale));
    const notDones = tf.tensor1d(batch.map(b => b.done ? 0 : 1));
    const weights = importanceWeights ? tf.tensor1d(importanceWeights) : tf.ones([batch.length]);




    const targetValues = tf.tidy(() => {
        const [nextMu, nextLogStd] = actor.predict(nextStates);
        const { action: nextActions, logProb: nextLogProb } = sampleAction(nextMu, nextLogStd);
        const nextStateActions = tf.concat([nextStates, nextActions], 1);

        const targetQ1 = targetCritic1.predict(nextStateActions).squeeze();
        const targetQ2 = targetCritic2.predict(nextStateActions).squeeze();
        const targetQ = tf.minimum(targetQ1, targetQ2);

        return rewards.add(notDones.mul(gamma).mul(targetQ.sub(nextLogProb.squeeze().mul(alpha))));
    });

    const stateActions = tf.concat([states, actions], 1);
    let criticLoss1, criticLoss2, tdErrors;


    const critic1Grads = optimizerCritic1.computeGradients(() => {
        const q1 = critic1.predict(stateActions).squeeze();
        const td1 = targetValues.sub(q1);
        tdErrors = tf.keep(td1.abs());
        return tf.losses.huberLoss(targetValues, q1, weights, 1, tf.Reduction.MEAN);
    });

    const clippedCritic1Grads = clipGradients(critic1Grads.grads, gradClip);
    optimizerCritic1.applyGradients(clippedCritic1Grads);
    criticLoss1 = critic1Grads.value;

    const critic2Grads = optimizerCritic2.computeGradients(() => {
        const q2 = critic2.predict(stateActions).squeeze();
        return tf.losses.huberLoss(targetValues, q2, weights, 1, tf.Reduction.MEAN);
    });
    const clippedCritic2Grads = clipGradients(critic2Grads.grads, gradClip);
    optimizerCritic2.applyGradients(clippedCritic2Grads);
    criticLoss2 = critic2Grads.value;

    // Only update actor every N steps (delayed policy updates)
    let actorLoss = tf.scalar(0);
    let alphaLoss = tf.scalar(0);
    let clippedActorGrads = null;
    let alphaGradsObj = null;

    if (CONFIG.trainSteps % actorDelay === 0) {
        const actorGrads = optimizerActor.computeGradients(() => {
            const [mu, logStd] = actor.predict(states);
            const { action: newAction, logProb } = sampleAction(mu, logStd);
            const stateNewActions = tf.concat([states, newAction], 1);

            const q1 = critic1.predict(stateNewActions).squeeze();
            const q2 = critic2.predict(stateNewActions).squeeze();
            const q = tf.minimum(q1, q2);
            const objective = q.sub(alpha.mul(logProb.squeeze()));
            return objective.mean().neg();
        });

        clippedActorGrads = clipGradients(actorGrads.grads, gradClip);
        optimizerActor.applyGradients(clippedActorGrads);
        actorLoss = actorGrads.value;

        const logProbAlpha = tf.tidy(() => {
            const [mu, logStd] = actor.predict(states);
            const { logProb } = sampleAction(mu, logStd);
            return logProb;
        });

        const alphaGradients = optimizerAlpha.computeGradients(() => {
            return logAlpha.mul(logProbAlpha.add(targetEntropy).squeeze()).mean().neg();
        });

        optimizerAlpha.applyGradients(alphaGradients.grads);
        alphaLoss = alphaGradients.value;
        alphaGradsObj = alphaGradients;
        
        tf.dispose(logProbAlpha);
    } else {
        log("Skipping actor update this step.");
        actorLoss = tf.scalar(0);
    }

    tf.dispose([states, nextStates, actions, rewards, notDones, weights, targetValues, stateActions]);
    tf.dispose(Object.values(clippedCritic1Grads));
    tf.dispose(Object.values(clippedCritic2Grads));
    if (clippedActorGrads) tf.dispose(Object.values(clippedActorGrads));
    if (alphaGradsObj) tf.dispose(alphaGradsObj.grads);

    const losses = {
        lossActor: actorLoss.dataSync()[0],
        lossCritic1: criticLoss1.dataSync()[0],
        lossCritic2: criticLoss2.dataSync()[0],
        tdErrors: tdErrors.arraySync(),
        alphaValue: alpha.dataSync()[0],
    };
    tf.dispose([tdErrors, actorLoss, criticLoss1, criticLoss2, alphaLoss]);

    return losses;
}