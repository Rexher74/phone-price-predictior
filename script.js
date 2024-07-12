/*  
    Tensorflow.js model that predicts the price range of phone between 0, 1, 2, 3 
    where 0 is the cheaper range and 3 is the most expensive.
*/

// Data path in CSV format
const TRAIN_DATA_PATH = "./train.csv"

// Declare the model, the max of the input data and the min of the input data (used in the normalization function) variables
var model, min, max;

/*  Function that returns an object with attributes INPUT and OUTPUT.
    INPUT is an array of arrays of type float containing the input parameters.
    OUTPUT is an array of type float containing the expected output for every input array.
*/
async function loadData(path) {
    // Get data and text
    let response = await fetch(path);
    let text = await response.text();

    // Separe the different rows
    let dataArray = text.split("\n");

    // Remove the headers row
    dataArray.shift();

    // Remove last row (that in the case of this data is empty)
    dataArray.pop();
    
    // Transfor the array of strings into an array of arrays of float
    for (let i = 0; i < dataArray.length; i++) {
        dataArray[i] = dataArray[i].split(",").map(e => parseFloat(e));
    }

    // Separe the last column into the output array
    let output = dataArray.map(dataArray => dataArray.pop());

    // Return the data
    return {INPUT: dataArray, OUTPUT: output};
}

// Retrurs the tensor normalized. In this case, this means all the values will be between 0 and 1.
function normalizeData(tensor, max, min) {
    /*  We obtain the result constant with a tf.tidy function. This means that it will automatically
        dispose all tensors used that are not returned at the end of the function, so we don't need to
        dispose them manually.
    */
    const result = tf.tidy(function() {
        // Find the minimum value contained in the tensor
        const MIN_VALUES = min || tf.min(tensor, 0);

        // Find the maximum value contained in the tensor
        const MAX_VALUES = max || tf.max(tensor, 0);

        // We Normalize the values of the tensor betwwen 0 and 1
        const TENSOR_SUBSTRACT_MIN_VAL =  tf.sub(tensor, MIN_VALUES);

        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

        const NORMALIZED_VALUES = tf.div(TENSOR_SUBSTRACT_MIN_VAL, RANGE_SIZE);

        return {NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES}
    })

    return result;
}

// Creates and trains a single layer sequential model with the inputs and outpus given
async function getModel(inputs, outputs) {
    // Create and define the model architecture
    const _model = tf.sequential();

    // We use 1 neuron (units) and an input of 20 feature values (inputs.shape[1])
    _model.add(tf.layers.dense({inputShape: [inputs.shape[1]], units: 1}));

    /*
    // Show the model features
    _model.summary()
    */

    /* TRAIN THE MODEL */

    // Define the learning rate suitable for our data 
    const LEARNING_RATE = 0.01;

    // Compile the model with the LEARNING_RATE and a loss function
    _model.compile({
        optimizer: tf.train.sgd(LEARNING_RATE),
        loss: "meanSquaredError"
    });

    // Do the training
    let results = await _model.fit(inputs, outputs, {
        validationSplit: 0.15, // 15% of data used for validation testing
        shuffle: true, // Shuffle inputs and outputs
        batchSize: 8, // Usually a 2^(something)
        epochs: 10  // Times to go over the data
    });

    inputs.dispose();
    outputs.dispose();

    // Check error
    console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
    console.log("Average validation error loss: " + Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));
    
    return _model;
}

// Function that returns the prediction of the input 'features' which is an array containing the featrures
export function predict(features) {
    if (!model || !min || !max) {
        // The model is still loading
        console.log("Model is still loading!");
        return false;
    }
    // Get the tensor from the input of the features
    const PREDICTION_INPUT = tf.tensor2d([features]);

    // Normalize the input
    let newInput = normalizeData(PREDICTION_INPUT, min, max);

    // Obtain the predicted output
    let output = model.predict(newInput.NORMALIZED_VALUES);

    return output;
}

// Disposes all the global tensors
function clearAll() {
    model.dispose();
    max.dispose();
    min.dispose();
}

export async function loadAll() {
    let data = await loadData(TRAIN_DATA_PATH);

    // Create the tensors objects
    const INPUTS_TENSOR = tf.tensor2d(data.INPUT);
    const OUTPUTS_TENSOR = tf.tensor1d(data.OUTPUT);

    // Obtain normalized tensor
    const RESULTS = normalizeData(INPUTS_TENSOR);

    // Save the max and min tensor values
    min = RESULTS.MIN_VALUES;
    max = RESULTS.MAX_VALUES;

    /*
    // Check Results
    console.log("Normalized values:");
    RESULTS.NORMALIZED_VALUES.print();
    console.log("Min Values:");
    RESULTS.MIN_VALUES.print();
    console.log("Max Values:");
    RESULTS.MAX_VALUES.print();
    */

    // We dispose the inputs tensor because we no longer need it as we have the normalized tensor
    INPUTS_TENSOR.dispose();

    getModel(RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR).then(resModel => model = resModel);

    // Now the model is ready to use by calling the function 'predict' defined above
}