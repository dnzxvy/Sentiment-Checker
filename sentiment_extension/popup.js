// when analyze button is clicked

document.getElementById("analyzeBtn").addEventListener("click", async () => {
	const text = document.getElementById("inputText").value;
	
	if (!text) {
		document.getElementById("result").innerText = "Please enter text.";
		return;
	}
	
	try {
	  // send text to backend api
	  const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",           // POST request
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })		// send { "text": " "}
	  });
	  
	  const data = await response.json(); // get json response
	  
	  // if prediction exists, display it
	  if (data.sentiment) {
		document.getElementById("result").innerText = "Sentiment: " + data.sentiment;
	  } else {
		
		// shows error if went wrong
		document.getElementById("result").innerText = "Error: " + JSON.stringify(data);
	  }
	
	 } catch (err) {
    // If backend is not running
    document.getElementById("result").innerText = "Error connecting to backend.";
  }
});
	 