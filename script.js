console.log('Hello tensorflow')

async function getData() {
  console.log('Getting car data')
  const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
  const carsData = await carsDataReq.json()
  console.log('Cleaning car data')
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower
  })).filter(car => (car.mpg !== null && car.horsepower !== null))

  console.log('Car data clean. Returning')

  return cleaned
}

function createModel() {
  console.log('Initializing model')
  const model = tf.sequential()

  console.log('Initializing input layer')
  model.add(tf.layers.dense({ inputShape: [1], units: 150, useBias: true }))
  model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }))
  model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }))
  model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }))
  model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }))
  model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }))
  model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }))
  console.log('Initializing oputput layer')
  model.add(tf.layers.dense({ units: 1, useBias: true }))

  return model
}

function convertToTensor(data) {
  return tf.tidy(() => {
    console.log('Shuffling data')
    tf.util.shuffle(data)

    console.log('Converting data to Tensor')
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg)

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [labels.length, 1])

    console.log('Normalizing data to range 0-1 using min-max scaling')
    const inputMax = inputTensor.max()
    const inputMin = inputTensor.min()
    const labelMax = labelTensor.max()
    const labelMin = labelTensor.min()

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin
    }

  })
}

async function trainModel(model, inputs, labels) {
  console.log('Preparing the model for training')
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  })

  const batchSize = 32;
  const epochs = 100

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  })
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData

  console.log('Generate predictions for a uniform range of numbers between 0 and 1')
  console.log('We unnormalize the data by doing the inverse of the min-max scaling')
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100)
    const preds = model.predict(xs.reshape([100, 1]))

    const unnormalizedXs = xs.mul(inputMax.sub(inputMin)).add(inputMin)

    const unnormalizedPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin)

    console.log('Unnormalizing data')
    return [unnormalizedXs.dataSync(), unnormalizedPreds.dataSync()]
  })

  const predictedPoints = Array.from(xs).map((val, i) => ({ x: val, y: preds[i] }))

  const originalPoints = inputData.map(d => ({ x: d.horsepower, y: d.mpg }))

  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  )
}

async function run() {
  console.log('Running main function')
  const data = await getData()

  console.log('Mapping data')
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }))

  console.log('Rendering scatterplot')
  tfvis.render.scatterplot(
    { name: 'Horsepower vs. MPG ' },
    { values },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  )
  console.log('Scatterplot rendered')

  console.log('Create model instance')
  const model = createModel()
  console.log('Displaying model summary')
  tfvis.show.modelSummary({ name: 'Model Summary' }, model)

  console.log('Trainig model prediction')
  const tensorData = convertToTensor(data)
  const { inputs, labels } = tensorData
  await trainModel(model, inputs, labels)
  console.log('Done training')
  testModel(model, data, tensorData)
  console.log('DONE')
}

document.addEventListener('DOMContentLoaded', run)
