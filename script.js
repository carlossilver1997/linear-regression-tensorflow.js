var stopTraining;
async function getData()  {
    const housesDataR = await fetch("data.json");
    const housesData = await housesDataR.json();
    let cleanData = housesData.map(house => ({
        price: house.price,
        rooms: house.avgRooms
    }));
    cleanData = cleanData.filter(house => (house.price != null && house.rooms != null));
    return cleanData;
};

const plotData = data => {
    const values = data.map(d => ({ x: d.rooms, y: d.price }));
    tfvis.render.scatterplot(
        { name: 'Rooms vs price' },
        { values },
        { 
            xLabel: 'Rooms', 
            yLabel: 'Rooms', 
            height: 300, 
        }
    );
};

const createModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [1], units: 1, useBias: true
    }));
    model.add(tf.layers.dense({ units: 1, useBias: true }));
    return model;
};

const optimizer = tf.train.adam();
const lossFunction = tf.losses.meanSquaredError;
const metrics = ['mse'];

const trainModel = async(model, inputs, labels) => {
    model.compile({
        optimizer: optimizer,
        loss: lossFunction,
        metrics: metrics
    });
    const surface = { name: 'show.history live', tab: 'Training' };
    const sizeBatch = 28;
    const epochs = 50;
    const history = [];

    return await model.fit(inputs, labels, {
        sizeBatch,
        epochs,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, log) => {
                history.push(log);
                tfvis.show.history(surface, history, ['loss', 'mse']);
                if(stopTraining){
                    model.stopTraining = true;
                }
            }
        }
    });
};

const transformDataToTensors = (data) => {
    return tf.tidy(() => {
        tf.util.shuffle(data);
        const inputs = data.map(d => d.rooms);
        const labels = data.map(d => d.price);

        const inputTensors = tf.tensor2d(inputs, [inputs.length,1]);
        const labelTensors = tf.tensor2d(labels, [labels.length,1]);

        const inputMax = inputTensors.max();
        const inputMin= inputTensors.min();
        const labelMax = labelTensors.max();
        const labelMin = labelTensors.min();

        // (dato-min) / (max-min)
        const inputNormalize = inputTensors
            .sub(inputMin)
            .div(inputMax.sub(inputMin));

        const labelNormalize = labelTensors
            .sub(labelMin)
            .div(labelMax.sub(labelMin));

        return {
            inputs: inputNormalize,
            labels: labelNormalize,
            inputMax,
            inputMin,
            labelMax,
            labelMin
        }
    });
};
var model;

const saveModel = async() => await model.save('downloads://regression-model');

const loadModel = async() => {
    const uploadJsonInput = document.getElementById('upload-json');
    const uploadWeightsInput = document.getElementById('upload-weights');
    model = await tf.loadLayersModel(tf.io.browserFiles([uploadJsonInput.files[0], uploadWeightsInput.files[0]]));
    console.log('Model loaded');
};

const showInferenceCurve = async() => {
    const data = await  getData();
    var tensorData = await transformDataToTensors(data);

    const { 
        inputMax, 
        inputMin, 
        labelMax, 
        labelMin
    } = tensorData;

    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));
        const desnormX = xs
            .mul(inputMax.sub(inputMin))
            .add(inputMin);

        const desnormY = preds
            .mul(labelMax.sub(labelMin))
            .add(labelMin);
        return [desnormX.dataSync(), desnormY.dataSync()];
    });

    const preditionPoints = Array.from(xs).map((val, i) => ({x: val, y: preds[i]}));
    const originalPoints = data.map(d => ({
        x: d.rooms, y: d.price
    }));
    tfvis.render.scatterplot(
        { 
            name: 'Predictions vs Originals' 
        },
        { 
            values: [originalPoints, preditionPoints], 
            series: ['originals', 'predictions']
        },
        {
            xLabel: 'Rooms',
            yLabel: 'Price',
            height: 300
        }
    );
};

async function  run ()  {
    const data = await getData();
    plotData(data);
    model  = createModel();
    const tensorData = transformDataToTensors(data);
    const { inputs, labels } = tensorData;
    await trainModel(model, inputs, labels);
};

run();