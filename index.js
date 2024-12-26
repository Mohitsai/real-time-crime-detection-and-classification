const mediaSelector = document.getElementById("media");
  
const webCamContainer =
    document.getElementById("web-cam-container");
  
let selectedMedia = "vid";
const video = document.querySelector('#myVidPlayer');

let chunks = [];
window.navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
            video.srcObject = stream;
            video.onloadedmetadata = (e) => {
                video.play();

                //new
                w = video.videoWidth;
                h = video.videoHeight

                canvas.width = w;
                canvas.height = h;
            };
        })
        .catch(error => {
            alert('Please enable the camera.');
        });  

  
function otherRecorderContainer(
    selectedMedia) {
  
    return selectedMedia
}
  

const audioMediaConstraints = {
    audio: true,
    video: false,
};

const videoMediaConstraints = {
  
 
    audio: true,
    video: true,
};
  
function startRecording(
    thisButton, otherButton) {
  
   
    navigator.mediaDevices.getUserMedia(
        selectedMedia === "vid" ? 
        videoMediaConstraints :
        audioMediaConstraints)
        .then((mediaStream) => {
  

        const mediaRecorder = 
            new MediaRecorder(mediaStream);
  
        window.mediaStream = mediaStream;

        window.mediaRecorder = mediaRecorder;
  
        mediaRecorder.start();
  
      
        mediaRecorder.ondataavailable = (e) => {
  
    
            chunks.push(e.data);
        };
  
      
        mediaRecorder.onstop = () => {
  
           
            const blob = new Blob(
                chunks, {
                    type:"video/mp4"
                });
            chunks = [];
            var cont =document.getElementById("id1");

           var item1=document.createElement("div" );
           item1.className = "item";
           cont.appendChild(item1);

           var div_ver = document.createElement("div" );
            div_ver.className = "div_ver";
            item1.appendChild(div_ver);

            var recordedMedia = document.createElement("video" );
            div_ver.appendChild(recordedMedia);
            recordedMedia.controls = true; 
            recordedMedia.className="rec";
           


            const recordedMediaURL = URL.createObjectURL(blob);  
            recordedMedia.src = recordedMediaURL;
            

            var downloadButton = document.createElement("a");
            div_ver.appendChild(downloadButton);
            downloadButton.download = "Recorded-Media";
            downloadButton.href = recordedMediaURL;
            downloadButton.innerText = "SAVE";
            
           
       URL.revokeObjectURL(recordedMedia);
        };
  
        if (selectedMedia === "vid") {

            webCamContainer.srcObject = mediaStream;
        }
  
        document.getElementById(
                `${selectedMedia}-record-status`)
                .innerText = "Recording";
  
        thisButton.disabled = true;
        otherButton.disabled = false;
    });
}
  
function stopRecording(thisButton, otherButton) {

    window.mediaRecorder.stop();
    window.mediaStream.getTracks()
    .forEach((track) => {
        track.stop();
    });
  
    document.getElementById(
            `${selectedMedia}-record-status`)
            .innerText = "Recording Stopped!";
    thisButton.disabled = true;
    otherButton.disabled = false;
}