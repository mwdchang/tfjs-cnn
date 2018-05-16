
class Model {
  constructor(learningRate) {
    this.learningRate = learningRate;
    /*
    this.model = tf.sequential();

    this.model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    this.model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    this.model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    this.model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    this.model.add(tf.layers.flatten());
    this.model.add(tf.layers.dense({units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));

    const optimizer = tf.train.sgd(this.learningRate);
    this.model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    */
  }

  build(layer1options, layer2options) {
    this.model = tf.sequential();

    // First layer
    this.model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: layer1options.kernelSize || 5,
      filters: layer1options.filters || 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    this.model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Second layer
    this.model.add(tf.layers.conv2d({
      kernelSize: layer2options.kernelSize || 5,
      filters: layer2options.filters || 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    this.model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Output layer
    this.model.add(tf.layers.flatten());
    this.model.add(tf.layers.dense({units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));

    const optimizer = tf.train.sgd(this.learningRate);
    this.model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  }


  predict(data) {
    return this.model.predict(data);
  }


  async train(data, numTrainBatches, callback=null) {
    const BATCH_SIZE = 64;
    // const TRAIN_BATCHES = 150;
    // const TRAIN_BATCHES = 40;
    const TRAIN_BATCHES = numTrainBatches;

    const TEST_BATCH_SIZE = 1000;
    const TEST_ITERATION_FREQUENCY = 5;

    for (let i = 0; i < TRAIN_BATCHES; i++) {
      const batch = data.nextTrainBatch(BATCH_SIZE);

      // Every few batches test the accuracy of the mode.
      let testBatch;
      let validationData;
      if (i % TEST_ITERATION_FREQUENCY === 0 || i === (TRAIN_BATCHES - 1)) {
        testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
        validationData = [
          testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
        ];
      }

      // The entire dataset doesn't fit into memory so we call fit repeatedly with batches.
      const history = await this.model.fit(
        batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
        {batchSize: BATCH_SIZE, validationData, epochs: 1});

      const loss = history.history.loss[0];
      const accuracy = history.history.acc[0];

      if (testBatch != null) {
        console.log('Batch ', i, ' accuracy', accuracy);
      }

      // Invoke callback
      if (callback) {
        callback(history);
      }

      // Clean up
      batch.xs.dispose();
      batch.labels.dispose();
      if (testBatch != null) {
        testBatch.xs.dispose();
        testBatch.labels.dispose();
      }
      await tf.nextFrame();
    }
  } // End train

}
