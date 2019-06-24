// initialization
var rec;
var recOpen = () => {
  var type = "wav";
  var bit = 16;
  var sample = 16000;
  rec = Recorder({
    type: type,
    bitRate: bit,
    sampleRate: sample
  });
  rec.open(() => {
    $("#recordbtn").removeClass("disabled");
  }, (e, isUserNotAllow) => {
    alert("Recording is not allowed! Please grant access!");
  });
}
recOpen();

// record
var recblob = {};
var startRecord = () => {
  if (rec) {
    rec.start();
    content = "<h4 class=\"mytheme\">录音中...</h4>";
    $("#content").empty();
    $("#content").html(content);
  }
  setTimeout(() => {
    stopRecord();
    $("#recordbtn").removeClass("disabled");
    $("#playbtn").removeClass("disabled");
    $("#content").empty();
  }, 2300);
}
var stopRecord = () => {
  if (rec) {
    rec.stop((blob, time) => {
      recblob = {blob: blob, time: time};
    }, (e) => {
      alert("record fail: " + e);
    });
  }
}
var record = () => {
  $("#recordbtn").addClass("disabled");
  $("#playbtn").addClass("disabled");
  $("#downloadbtn").addClass("disabled");
  $("#predictbtn").addClass("disabled");
  startRecord();
}

// play
var play = () => {
  if (recblob) {
    console.log(recblob.time)
    var audio = $("#recplay")[0];    
    audio.controls = true;
    audio.src = URL.createObjectURL(recblob.blob);
    audio.play();
    // $("#downloadbtn").removeClass("disabled");
    $("#predictbtn").removeClass("disabled");
  }
}


// predict
var predict = () => {
  if (recblob) {
    var form = new FormData();
    form.append("upfile", recblob.blob, "1.wav");
    $.ajax({
      url: '/predict',
      type: "POST",
      contentType: false,
      processData: false,
      data: form,
      success: (data) => {
        content = "<h4 class=\"mytheme\">" + data + "</h4>";
        $("#content").empty();
        $("#content").html(content);
      },
      error: (e) => {
        alert("download failed!");
      }
    });
  }
}

