normalPoints = [];
anomalyPoints = [];

drawingNormal = true;

function setup() {
  createCanvas(1024, 1024);
  background(255);
  frameRate(1000);
}

function draw() { 
  if (mouseIsPressed === true) { 
    
    currX = mouseX;
    currY = mouseY;
    mappedX = map(currX, 0, width, -10, 10);
    mappedY = map(currY, 0, height, -10, 10);   
    
    if (drawingNormal) {
      stroke('blue');
      strokeWeight(10);
      point(currX, currY);
      normalPoints.push([mappedX ,mappedY])
    } else {
      stroke('red');
      strokeWeight(10);
      point(currX, currY);
      anomalyPoints.push([mappedX ,mappedY])
    }
  }  
} 

function keyPressed() {
  if (keyCode === LEFT_ARROW) {
    drawingNormal = true;
    print('drawing normal')
  } else if (keyCode === RIGHT_ARROW) {
    drawingNormal = false;
    print('drawing anomaly')
  } else if (keyCode === ENTER){
    noLoop();
    saveData();
  } 
  print("Number of normal points: " + normalPoints.length);
  print("Number of anomal points: " + anomalyPoints.length);
}

function saveData(){
    json = {"normal": normalPoints, "anomaly": anomalyPoints};
    saveJSON(json);
}