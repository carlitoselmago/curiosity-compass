self.importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js");

let procesimgsize=64;
let params = 32;
let size = 6;
let size2 = 2//4;
let startTime = new Date();
let mintimebetwenfit_seconds=0.3;

const model = tf.sequential();

// Encoder
model.add(tf.layers.conv2d({
    filters: params,
    kernelSize: [size, size],
    activation: 'relu',
    padding: 'same',
    inputShape: [procesimgsize, procesimgsize, 1]
}));
model.add(tf.layers.maxPooling2d({
    poolSize: [size2, size2],
    padding: 'same'
}));
model.add(tf.layers.conv2d({
    filters: params,
    kernelSize: [size, size],
    activation: 'relu',
    padding: 'same'
}));
model.add(tf.layers.maxPooling2d({
    poolSize: [size2, size2],
    padding: 'same'
}));

// Decoder
model.add(tf.layers.conv2d({
    filters: params,
    kernelSize: [size, size],
    activation: 'relu',
    padding: 'same'
}));
model.add(tf.layers.upSampling2d({
    size: [size2, size2]
}));
model.add(tf.layers.conv2d({
    filters: params,
    kernelSize: [size, size],
    activation: 'relu',
    padding: 'same'
}));
model.add(tf.layers.upSampling2d({
    size: [size2, size2]
}));
model.add(tf.layers.conv2d({
    filters: 1,
    kernelSize: [size, size],
    activation: 'sigmoid',
    padding: 'same'
}));

const compileArgs = {
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy'  
};

model.compile(compileArgs);

self.onmessage= async (e) => {
    let imageData=e.data;
  
    // Convert the image data to a tensor
    let tensor = tf.browser.fromPixels(imageData, 1).toFloat().expandDims();

    // Resize it to model's required input size and convert to grayscale
    tensor = tf.image.resizeBilinear(tensor, [procesimgsize, procesimgsize]);
    tensor = tensor.mean(3).expandDims(3);

    // Normalization - Rescale pixel values from [0, 255] to [0, 1]
    tensor = tensor.div(tf.scalar(255));
    
    const decoded_image =  await model.predict(tensor); 

    // Reshape decoded_image to shape of original image
    const decoded_image_reshaped = decoded_image.reshape(tensor.shape);

    // Calculate mean squared error
    //const mse = tf.metrics.meanSquaredError(image, decoded_image_reshaped);
    let mse = tf.tidy(() => {
      let diff = tensor.sub(decoded_image_reshaped);
      let squarred = diff.square();   
      return squarred.mean();
    });
    mse=  await mse.data();
    let avg_mse=mse[0]*2000;
    
    //don't fit each iteration
   
    let endTime = new Date();
    let timeElapsed = endTime - startTime;
    timeElapsed= ((timeElapsed % 60000) / 1000).toFixed(0);
    if (timeElapsed > mintimebetwenfit_seconds){
         //fit model 
        await  model.fit(tensor, tensor, {epochs:1});
        startTime=new Date();
    } 
    
   
    postMessage(avg_mse.toFixed(2));
}

