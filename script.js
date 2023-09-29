//tf.setBackend('cpu');
//TS MODEL
// Procesimgsize needs its value here.
let procesimgsize=64;
let params = 32;
let size = 6;
let size2 = 2//4;
/*
const model = tf.sequential();
model.add(tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', inputShape: [64, 64, 1]}));
model.add(tf.layers.conv2d({filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu'}));

// Decoder
model.add(tf.layers.conv2dTranspose({filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu'}));
model.add(tf.layers.conv2dTranspose({filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu'}));
model.add(tf.layers.conv2d({filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid'}));

model.compile({optimizer: 'adam', loss: 'binaryCrossentropy'});
//console.log(model.summary());
*/
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

//let tf = require('@tensorflow/tfjs-node');
var video = document.querySelector("video");
var canvas = document.querySelector("canvas");
var ctx = canvas.getContext('2d');

if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({video: true})
  .then(function(stream) {
    video.srcObject = stream;
    video.onloadedmetadata = function(e) {
      video.play();
    };
  })
  .catch(function(videoError) {
    console.error(videoError);
  });
}

async function processVideoFrame() {
  
  if (!video.paused && !video.ended) {

      // Get the video width and height as it might not be 640x480
      let width = video.videoWidth;
      let height = video.videoHeight;

      // Calculate image scale
      let scale = Math.min(procesimgsize / width, procesimgsize / height);

      // Calculate scaled image size
      let newWidth = Math.floor(width * scale);
      let newHeight = Math.floor(height * scale);

      // Adjust canvas size
      canvas.width = newWidth;
      canvas.height = newHeight;

      //ctx.filter = 'blur(10px)';

      // Draw the video frame to canvas (scaled down)
      ctx.drawImage(video, 0, 0, newWidth, newHeight);

      // Get all image data from the resized frame
      var imageData = ctx.getImageData(0, 0, newWidth, newHeight);
      
      // Convert the image data to a tensor
      let tensor = tf.browser.fromPixels(imageData, 1).toFloat().expandDims();

      // Resize it to model's required input size and convert to grayscale
      tensor = tf.image.resizeBilinear(tensor, [procesimgsize, procesimgsize]);
      tensor = tensor.mean(3).expandDims(3);
      
      // Pass the tensor to your prediction function
      await predict_and_calculate_mse(tensor);
  }

  // Set up the next frame to be processed
  requestAnimationFrame(processVideoFrame);
}

// Start the video processing
processVideoFrame();
/*
function draw(video, ctx, width, height) {
    
    if (video.paused || video.ended) return false;

        // Resize the canvas to match the video's native resolution
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

    // Else, draw video frame on canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    // You can add any specific processing code here

    // Continue drawing next frame
    requestAnimationFrame(() => draw(video, ctx, width, height));
}
*/
function videoError(e) {
  // You can put some error handling code here
  console.log('Webcam error!', e);
};



async function predict_and_calculate_mse(image) {
  try {
    if (image){
      const decoded_image = await model.predict(image); 

      // Reshape decoded_image to shape of original image
      const decoded_image_reshaped = decoded_image.reshape(image.shape);

      // Calculate mean squared error
      //const mse = tf.metrics.meanSquaredError(image, decoded_image_reshaped);
      let mse = tf.tidy(() => {
        let diff = image.sub(decoded_image_reshaped);
        let squarred = diff.square();
        return squarred.mean();
      });
      mse=  await mse.data();
      let avg_mse=mse[0]/2000
      //let mseval=mse.dataSync()
      //let avg_mse = mse.mean().dataSync()[0]/2000;
      console.log(avg_mse);
      //console.log(mseval);

      //fit model 
      await model.fit(image, image, {epochs:5});

     // return mseval;
    }
 } catch (e) {
      console.error("predict_and_calculate_mse EXCEPTION!!!:", e);
      //this.end(); // If this is defined somewhere in your JS object
  }
}