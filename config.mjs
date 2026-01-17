export const CONFIG = {
    ACTOR_LAYER_LENGTHS: [256, 256],
    CRITIC_LAYER_LENGTHS: [256, 256],
    GAME_STATE_SIZE: 28,
    ACTION_SIZE: 6,

    LEARNING_RATE_ACTOR: 0.0001,
    LEARNING_RATE_CRITIC: 0.0003,
    
    SAVE_AFTER_EPISODES: 300,
    BATCH_SIZE: 256,
    DISCOUNT_FACTOR: 0.99,
    REPLAY_BUFFER_SIZE: 500_000,

    TARGET_UPDATE_FREQ: 20,
    TRAIN_FREQ: 50,

    // Prioritized Experience Replay parameters
    PER_ALPHA: 0.6,       // Priority exponent (0 = uniform, 1 = full prioritization)
    PER_BETA_START: 0.4,  // Initial importance sampling exponent
    PER_BETA_END: 1.0,    // Final importance sampling exponent
    PER_BETA_FRAMES: 100_000, // Number of frames to anneal beta
    TEMP_START: 0.5,
    TEMP: 0.5,
    TEMP_END: 0.1,
    TEMP_ANNEAL_FRAMES: 100_000,

    ENTROPY_COEFFICIENT: 0.3,

    PLAYER_ONE_ID: 200,
    PLAYER_TWO_ID: 201,
    POSITION_NORMALIZATION: 1 / 500,
    VELOCITY_NORMALIZATION: 1.5,

    // runtime params
    trainSteps: 0
};

export function updateConfig(newConfig) {
    Object.assign(CONFIG, newConfig);
}

top.CONFIG = CONFIG;