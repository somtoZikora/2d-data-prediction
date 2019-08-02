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
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
  console.log('Initializing oputput layer')
  model.add(tf.layers.dense({ units: 1, useBias: true }))

  return model
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
}

document.addEventListener('DOMContentLoaded', run)
