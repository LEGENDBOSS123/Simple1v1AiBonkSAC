import { CONFIG, updateConfig } from "./config.mjs";
import { log } from "./log.mjs";
import { actionToArray, arrayToAction, move, predictActionArray } from "./move.mjs";
import { Random } from "./Random.mjs";
import { ReplayBuffer } from "./ReplayBuffer.mjs";
import { setupLobby } from "./setupLobby.mjs";
import { cloneModel, deserializeModels, loadBrowserFile, setupModel } from "./setupModel.mjs";
import { State } from "./State.mjs";
import { tf } from "./tf.mjs";
import { Time } from "./Time.mjs";
import { train } from "./train.mjs";

let models = [];
let currentModel = null;
let memory = new ReplayBuffer(CONFIG.REPLAY_BUFFER_SIZE);

let losses = [];

top.models = function () { return models; };
top.currentModel = function () { return currentModel; };
top.memory = function () { return memory; };
top.losses = function () { return losses; };
top.paused = false;

async function setup() {
    const filePrompt = prompt("Do you want to load a model from file? (y/n)", "n") == "y";
    const filedata = filePrompt ? await loadBrowserFile() : null;
    if (filedata) {
        updateConfig(filedata.CONFIG);
        const deserialized = await deserializeModels(filedata.models);
        models = deserialized.models;
        currentModel = deserialized.currentModel;
        memory = ReplayBuffer.fromJSON(filedata.per);
        log(`Loaded ${models.length} models from file.`);
    }
    else {
        currentModel = setupModel();
    }
    log("TensorFlow.js version:", tf.version.tfjs);
    log(`Actor Model initialized with ${currentModel.actor.countParams()} parameters.`);
    log(`Critic Model initialized with ${currentModel.critic1.countParams()} parameters.`);
}


async function main() {

    await setupLobby();

    log("Lobby setup complete.");


    async function gameLoop() {

        while (true) {


            if (memory.size() >= CONFIG.BATCH_SIZE) {
                log("Training...");
                // Anneal beta from PER_BETA_START to PER_BETA_END over PER_BETA_FRAMES
                const beta = Math.min(
                    CONFIG.PER_BETA_END,
                    CONFIG.PER_BETA_START + (CONFIG.PER_BETA_END - CONFIG.PER_BETA_START) * (CONFIG.trainSteps / CONFIG.PER_BETA_FRAMES)
                );
                CONFIG.TEMP = Math.max(
                    CONFIG.TEMP_END,
                    CONFIG.TEMP_START - (CONFIG.TEMP_START - CONFIG.TEMP_END) * (CONFIG.trainSteps / CONFIG.TEMP_ANNEAL_FRAMES)
                );
                for (let i = 0; i < CONFIG.TRAIN_FREQ; i++) {
                    const { batch, indices, importanceWeights } = memory.sample(CONFIG.BATCH_SIZE, CONFIG.PER_ALPHA, beta);
                    const result = await train(currentModel, batch, importanceWeights);
                    if (result) {
                        losses.push(result.lossActor);
                        log(`Loss: ${result.lossActor.toFixed(4)}`);
                        log(`Critic1 Loss: ${result.lossCritic1.toFixed(4)}`);
                        log(`Critic2 Loss: ${result.lossCritic2.toFixed(4)}`);
                        CONFIG.trainSteps ++;

                        if (CONFIG.trainSteps % (CONFIG.TARGET_UPDATE_FREQ * CONFIG.TRAIN_FREQ) === 0) {
                            currentModel.targetCritic1.setWeights(currentModel.critic1.getWeights());
                            currentModel.targetCritic2.setWeights(currentModel.critic2.getWeights());
                            log("Target network updated.");
                        }
                    }
                    memory.updatePriorities(indices, result.tdErrors);
                }
                // Save model checkpoint after training batch
                if (CONFIG.trainSteps % CONFIG.SAVE_AFTER_EPISODES === 0 && CONFIG.trainSteps > 0) {
                    models.push(cloneModel(currentModel));
                    if (models.length > 10) {
                        models.shift();
                    }
                    log(`Model checkpoint saved. Total checkpoints: ${models.length}`);
                }
            }

            // match start
            top.startGame();
            await Time.sleep(1500);

            // 20 TPS
            let TPS = 1000 / 20;

            let lastState = new State();
            lastState.fetch();
            await Time.sleep(TPS);

            let newState;

            let safeFrames = 0;

            let p2Model = currentModel;
            if (Math.random() < 0.5 && models.length > 1) {
                p2Model = Random.choose(models);
            }

            let lastActionP1 = null;
            let lastActionP2 = null;
            while (true) {

                newState = new State();
                newState.fetch();

                let rewardCurrentFrame = newState.reward();
                let rewardP1 = rewardCurrentFrame.p1;
                let rewardP2 = rewardCurrentFrame.p2;

                if (safeFrames > 300) {
                    newState.done = true;
                }

                if (lastActionP1 !== null) {
                    memory.add(lastState.toArray(),
                        lastActionP1,
                        rewardP1,
                        newState.toArray(),
                        newState.done
                    );
                }

                if (lastActionP2 !== null) {
                    memory.add(lastState.flip().toArray(),
                        lastActionP2,
                        rewardP2,
                        newState.flip().toArray(),
                        newState.done
                    );
                }

                if (newState.done) {
                    break;
                }

                let probs = predictActionArray(currentModel.actor, newState.toArray());
                top.probs = probs;
                let ataP1 = arrayToAction(probs);
                move(CONFIG.PLAYER_ONE_ID, ataP1);
                lastActionP1 = actionToArray(ataP1);


                let probs2 = predictActionArray(p2Model.actor, newState.flip().toArray());
                let ataP2 = arrayToAction(probs2);
                move(CONFIG.PLAYER_TWO_ID, ataP2);
                lastActionP2 = actionToArray(ataP2);

                safeFrames++;
                lastState = newState;

                // 20 FPS
                await Time.sleep(50);
            }

            if (top.paused) {
                log("Paused. Press OK to continue.");
                top.paused = false;
                return;
            }
        }
    }

    gameLoop();
}

function pause() {
    top.paused = true;
}

top.main = main;
setup().then(() => { main() });