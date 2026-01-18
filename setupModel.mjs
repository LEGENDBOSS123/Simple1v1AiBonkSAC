import { CONFIG } from "./config.mjs";
import { tf } from "./tf.mjs";


function setupSACActor() {
    const input = tf.input({ shape: [CONFIG.GAME_STATE_SIZE], name: 'actor_input' });

    let x = input;
    let i = 0;

    for (const layerLength of CONFIG.ACTOR_LAYER_LENGTHS) {
        x = tf.layers.dense({
            units: layerLength,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: `actor_dense_${i++}`
        }).apply(x);
    }
    const mu = tf.layers.dense({
        units: CONFIG.ACTION_SIZE,
        activation: 'linear',
        name: 'actor_mean'
    }).apply(x);
    const logStd = tf.layers.dense({
        units: CONFIG.ACTION_SIZE,
        activation: 'linear',
        name: 'actor_log_std'
    }).apply(x);

    return tf.model({ inputs: input, outputs: [mu, logStd] });
}

function setupSACCritic() {
    let model = tf.sequential();
    model.add(tf.layers.inputLayer({ inputShape: [CONFIG.GAME_STATE_SIZE + CONFIG.ACTION_SIZE], name: 'critic_input' }));
    let i = 0;
    for (const layerLength of CONFIG.CRITIC_LAYER_LENGTHS) {
        model.add(tf.layers.dense({
            units: layerLength,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: `critic_dense_${i++}`
        }));
    }
    model.add(tf.layers.dense({
        units: 1,
        activation: 'linear',
        kernelInitializer: 'heNormal',
        name: 'critic_output'
    }));
    return model;
}

export function setupModel() {
    const actor = setupSACActor();
    const critic1 = setupSACCritic();
    const critic2 = setupSACCritic();
    const targetCritic1 = setupSACCritic();
    const targetCritic2 = setupSACCritic();
    targetCritic1.setWeights(critic1.getWeights());
    targetCritic2.setWeights(critic2.getWeights());
    const logAlpha = tf.variable(tf.scalar(0), true);
    const optimizerAlpha = tf.train.adam(CONFIG.LEARNING_RATE_ACTOR);
    const optimizerActor = tf.train.adam(CONFIG.LEARNING_RATE_ACTOR);
    const optimizerCritic1 = tf.train.adam(CONFIG.LEARNING_RATE_CRITIC);
    const optimizerCritic2 = tf.train.adam(CONFIG.LEARNING_RATE_CRITIC);
    return {
        actor,
        critic1,
        critic2,
        targetCritic1,
        targetCritic2,
        optimizerActor,
        optimizerCritic1,
        optimizerCritic2,
        logAlpha,
        optimizerAlpha
    };
}


export function cloneModel(model) {
    const newModel = setupModel();
    newModel.actor.setWeights(model.actor.getWeights());
    newModel.critic1.setWeights(model.critic1.getWeights());
    newModel.critic2.setWeights(model.critic2.getWeights());
    newModel.targetCritic1.setWeights(model.targetCritic1.getWeights());
    newModel.targetCritic2.setWeights(model.targetCritic2.getWeights());
    newModel.logAlpha.assign(model.logAlpha);
    return newModel;
}

export async function downloadModel(model) {
    const artifacts = await model.save(
        tf.io.withSaveHandler(async (artifacts) => {
            return artifacts;
        })
    );
    artifacts.weightData = Array.from(new Float32Array(artifacts.weightData));
    return artifacts;
}

export async function serializeModels(models, currentModel) {
    if (currentModel) {
        models.push(currentModel);
    }
    const arrayOfModels = models;

    const serializedModels = [];
    for (const model of arrayOfModels) {
        const artifactsActor = await downloadModel(model.actor);
        const artifactsCritic1 = await downloadModel(model.critic1);
        const artifactsCritic2 = await downloadModel(model.critic2);
        const artifactsTargetCritic1 = await downloadModel(model.targetCritic1);
        const artifactsTargetCritic2 = await downloadModel(model.targetCritic2);
        const artifacts = {
            actor: artifactsActor,
            critic1: artifactsCritic1,
            critic2: artifactsCritic2,
            targetCritic1: artifactsTargetCritic1,
            targetCritic2: artifactsTargetCritic2,
            logAlpha: Array.from(model.logAlpha.dataSync())
        };
        serializedModels.push(artifacts);
    }
    return serializedModels;
}

top.saveModels = async function () {
    const serializedModels = await serializeModels(top.models(), top.currentModel());
    const json = {
        models: serializedModels,
        per: top.memory().toJSON(),
        CONFIG: CONFIG
    }
    await saveBrowserFile(json, "models.json");
}

export async function loadModelFromArtifacts(artifacts) {
    const model = setupModel();

    model.actor.loadWeights(tf.io.decodeWeights(new Float32Array(artifacts.actor.weightData).buffer, artifacts.actor.weightSpecs));
    model.critic1.loadWeights(tf.io.decodeWeights(new Float32Array(artifacts.critic1.weightData).buffer, artifacts.critic1.weightSpecs));
    model.critic2.loadWeights(tf.io.decodeWeights(new Float32Array(artifacts.critic2.weightData).buffer, artifacts.critic2.weightSpecs));
    model.targetCritic1.loadWeights(tf.io.decodeWeights(new Float32Array(artifacts.targetCritic1.weightData).buffer, artifacts.targetCritic1.weightSpecs));
    model.targetCritic2.loadWeights(tf.io.decodeWeights(new Float32Array(artifacts.targetCritic2.weightData).buffer, artifacts.targetCritic2.weightSpecs));
    console.log(artifacts.logAlpha);
    model.logAlpha.assign(tf.tensor(artifacts.logAlpha[0]));

    return model;
}

export async function deserializeModels(arrayOfArtifacts) {
    const models = [];
    for (const artifacts of arrayOfArtifacts) {
        const model = await loadModelFromArtifacts(artifacts);
        models.push(model);
    }
    const currentModel = models.pop();
    return { models, currentModel };
}


export async function saveBrowserFile(filedata, filename) {
    // 1. Convert JSON to string
    const jsonString = JSON.stringify(filedata);

    // 2. Create a stream from the string data
    const stream = new Blob([jsonString]).stream();

    // 3. Pipe through GZIP compression
    const compressedStream = stream.pipeThrough(new CompressionStream("gzip"));

    // 4. Convert stream back to a Blob
    const compressedBlob = await new Response(compressedStream).blob();

    // 5. Download logic (Standard)
    const url = URL.createObjectURL(compressedBlob);
    const a = document.createElement("a");
    a.href = url;
    // Append .gz to indicate compression, or keep original if you handle extension logic elsewhere
    a.download = filename.endsWith('.gz') ? filename : filename + '.gz';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}

/**
 * Loads a GZIP compressed file from the user and parses it back to JSON.
 */
export function loadBrowserFile() {
    return new Promise((resolve, reject) => {
        const input = document.createElement("input");
        input.type = "file";
        // Accept .json or .gz files
        input.accept = "application/json, .gz";

        input.onchange = async (event) => {
            const file = event.target.files[0];
            if (!file) return;

            try {
                // 1. Get the stream from the file
                const stream = file.stream();

                // 2. Pipe through GZIP decompression
                const decompressedStream = stream.pipeThrough(new DecompressionStream("gzip"));

                // 3. Read the decompressed stream as text
                const text = await new Response(decompressedStream).text();

                // 4. Parse JSON
                const filedata = JSON.parse(text);
                resolve(filedata);
            } catch (error) {
                console.error("Decompression failed. Was the file actually compressed?", error);
                // Fallback: Try reading as plain JSON in case the user uploaded an uncompressed file
                try {
                    const text = await file.text();
                    resolve(JSON.parse(text));
                } catch (fallbackError) {
                    reject(error);
                }
            }
        };

        input.click();
    });
}