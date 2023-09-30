onmessage = function(e) {
    console.log(e.data)
};
/*
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
    //console.log(avg_mse);
    //$("#data").html(avg_mse.toFixed(2));
    //console.log(mseval);

    //fit model 
    await model.fit(image, image, {epochs:5});
    postMessage(avg_mse.toFixed(2));
   // return mseval;
} else {
    postMessage(0);
}
*/