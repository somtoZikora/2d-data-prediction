console.log('Hello tensorflow')

/**
 * Section 3: Load format and visualise the input data
 * Get the car data reduced to just the variables we're interested in
 * and cleaned of missing data
 */

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
}

document.addEventListener('DOMContentLoaded', run)
