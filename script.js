//tf.setBackend('cpu');

// Procesimgsize needs its value here.
let procesimgsize=64;

window.working=false;

var video = document.querySelector("video");
var canvas = document.querySelector("canvas");
var ctx = canvas.getContext('2d',{willReadFrequently: true});

navigator.mediaDevices.getUserMedia = navigator.mediaDevices.getUserMedia ||
navigator.webkitGetUserMedia ||
navigator.mozGetUserMedia;

if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({video: true})
  .then(function(stream) {
    video.srcObject = stream;
    video.onloadedmetadata = function(e) {
      video.play();
    };
  })
  .catch(function(videoError) {
   alert("Camera not available");
  }); 
}

let worker = new Worker("curiosity_worker.js");//,{type: 'module'});

worker.onmessage = function(event){
  $("#data").html(event.data);
  window.working=false;
};

async function runWorker(imageData){
  window.working=true;
 // console.log("run worker");
  worker.postMessage(imageData);
}

async function processVideoFrame() {

    if (!video.paused && !video.ended ) {

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
        
        // Pass the tensor to your prediction function
        if (!window.working){
          await runWorker(imageData);
        } else { 
          //console.log("worker busy");
        }
        //await predict_and_calculate_mse(tensor);
    }
    // Set up the next frame to be processed
    requestAnimationFrame(processVideoFrame);
  
}

// Start the video processing
processVideoFrame();

function videoError(e) {
  // You can put some error handling code here
  console.log('Webcam error!', e);
};
