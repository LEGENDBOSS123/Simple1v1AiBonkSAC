import { CONFIG } from "./config.mjs";
import { tf } from "./tf.mjs";


export async function train(models, batch, importanceWeights = null) {
    if (batch.length < CONFIG.BATCH_SIZE) return null;

    const { actor, critic1, critic2, targetCritic1, targetCritic2, optimizerActor, optimizerCritic1, optimizerCritic2 } = models;
    const gamma = CONFIG.DISCOUNT_FACTOR;
    const entropyCoefficient = CONFIG.ENTROPY_COEFFICIENT;
    const eps = 1e-8;
    const clip = 0.01;
    const temp = CONFIG.TEMP;

    const states = tf.tensor2d(batch.map(b => b.state));
    const nextStates = tf.tensor2d(batch.map(b => b.nextState));
    const actions = tf.tensor2d(batch.map(b => b.action));
    const rewards = tf.tensor1d(batch.map(b => b.reward));
    const notDones = tf.tensor1d(batch.map(b => b.done ? 0 : 1));
    const weights = importanceWeights
        ? tf.tensor1d(importanceWeights)
        : tf.ones([batch.length]);




    const targetValues = tf.tidy(() => {
        const nextProbs = actor.predict(nextStates).clipByValue(clip, 1 - clip);

        // sigmoid((log(p) - log(1-p) + log(u) - log(1-u)) / temp)
        const logits = nextProbs.add(eps).log().sub(tf.scalar(1).sub(nextProbs).add(eps).log());
        const u = tf.randomUniform(logits.shape);
        const gumbel = u.add(eps).log().add(tf.scalar(1).sub(u).add(eps).log().neg());
        const nextActions = logits.add(gumbel).div(temp).sigmoid();
        const nextStateActions = tf.concat([nextStates, nextActions], 1);
        const targetQ1 = targetCritic1.predict(nextStateActions).squeeze();
        const targetQ2 = targetCritic2.predict(nextStateActions).squeeze();
        const targetQ = tf.minimum(targetQ1, targetQ2);

        // -sum(p*log(p) + (1-p)*log(1-p))
        const term1 = nextProbs.add(eps).log().mul(nextActions);
        const term2 = tf.scalar(1).sub(nextProbs).add(eps).log().mul(tf.scalar(1).sub(nextActions));
        const logProb = term1.add(term2).sum(1);

        // r + gamma * (1 - done) * (Q_target - alpha * H)
        return rewards.add(notDones.mul(gamma).mul(targetQ.sub(logProb.mul(entropyCoefficient))));
    });

    const stateActions = tf.concat([states, actions], 1);
    let criticLoss1, criticLoss2, tdErrors, actorLoss;


    criticLoss1 = optimizerCritic1.minimize(() => {
        const q1 = critic1.predict(stateActions).squeeze();
        const td1 = targetValues.sub(q1);
        tdErrors = tf.keep(td1.abs());
        return tf.losses.huberLoss(targetValues, q1, weights, 1, tf.Reduction.MEAN);
    }, true);


    criticLoss2 = optimizerCritic2.minimize(() => {
        const q2 = critic2.predict(stateActions).squeeze();
        return tf.losses.huberLoss(targetValues, q2, weights, 1, tf.Reduction.MEAN);
    }, true);

    actorLoss = optimizerActor.minimize(() => {
        const probs = actor.predict(states).clipByValue(clip, 1 - clip);


        // sigmoid((log(p) - log(1-p) + log(u) - log(1-u)) / temp)
        const logits = probs.add(eps).log().sub(tf.scalar(1).sub(probs).add(eps).log());
        const u = tf.randomUniform(logits.shape);
        const gumbel = u.add(eps).log().add(tf.scalar(1).sub(u).add(eps).log().neg());

        const expectedActions = logits.add(gumbel).div(temp).sigmoid();
        const stateActions = tf.concat([states, expectedActions], 1);

        const q1 = critic1.predict(stateActions).squeeze();
        const q2 = critic2.predict(stateActions).squeeze();
        const q = tf.minimum(q1, q2);

        // H = -sum(p*log(p) + (1-p)*log(1-p))
        const term1 = probs.add(eps).log().mul(expectedActions);
        const term2 = tf.scalar(1).sub(probs).add(eps).log().mul(tf.scalar(1).sub(expectedActions));
        const logProb = term1.add(term2).sum(1);

        // Actor loss: minimize -Q + alpha * (-H) = maximize Q + alpha * H
        const loss = logProb.mul(entropyCoefficient).sub(q).mean();
        return loss;
    }, true);

    tf.dispose(
        [states, nextStates, actions, rewards, notDones, weights,
            targetValues, stateActions]
    );

    const losses = {
        lossActor: actorLoss.dataSync()[0],
        lossCritic1: criticLoss1.dataSync()[0],
        lossCritic2: criticLoss2.dataSync()[0],
        tdErrors: tdErrors.arraySync()
    };
    tf.dispose([tdErrors, actorLoss, criticLoss1, criticLoss2]);

    return losses;
}