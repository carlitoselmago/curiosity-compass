//tf.setBackend('cpu');

//HELPER FUNCTIONS
function map(input,in_min, in_max, out_min, out_max) {
  let res= (input - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
 // if (res>out_max){res=out_max;}
  return res;
}

var lerp = function (oldValue, newValue, factor) {
  return oldValue * (1 - factor) + newValue * factor;
}

// Procesimgsize needs its value here.
let procesimgsize=64;
let Cvalue=0; //curiosity value
let Cvalue_old=0; // curiosity last value
let itpl_factor=0.2 // interpolate factor

window.working=false;

var video = document.querySelector("video");
var canvas = document.querySelector("canvas");
var ctx = canvas.getContext('2d',{willReadFrequently: true});

navigator.mediaDevices.getUserMedia = navigator.mediaDevices.getUserMedia ||
navigator.webkitGetUserMedia ||
navigator.mozGetUserMedia;

var currentStream;
var isFrontCamera = true;

function stopCurrentStream(){
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }
}

function toggleCamera(){
  isFrontCamera = !isFrontCamera;
  stopCurrentStream();
  var constraints = { video: { facingMode: (isFrontCamera? "user" : "environment") } };
  
  navigator.mediaDevices.getUserMedia(constraints)
    .then(function(stream) {
      currentStream = stream;
      video.srcObject = stream;
      video.onloadedmetadata = function(e) {
        video.play();
      };
    })
    .catch(function(err) {
      console.log(err.name + ": " + err.message);
    });
}

//initial setup
toggleCamera();

$("body").click(function(e){
  toggleCamera();
})

let worker = new Worker("curiosity_worker.js");//,{type: 'module'});

worker.onmessage = function(event){
  let value=event.data;
  
  Cvalue=map(value,0, 30, 0, 100);
  
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

        //UPDATE UI elements
        let interpolatedvalue= lerp(Cvalue_old, Cvalue, itpl_factor); 
      // $("#data").html(value);
          $("#circle svg").css("transform","scale("+interpolatedvalue+")");
          Cvalue_old= interpolatedvalue;

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


