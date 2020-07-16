#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Dashboard for Video-based Analytics for Livestock Farming

#import packages
import copy
import dash
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import cv2 as cv
from tensorflow.keras.models import load_model
import pickle
import cv2
import plotly.graph_objects as go
import dash_player

# create the app object
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

VIDEO_PATH = "assets/sample3.mp4"
#VIDEO_PATH = "assets/N190918.mp4"
model = load_model('model/model.h5')
lb = pickle.loads(open('model/lb.pickle', "rb").read())

TIME_SKIPS = float(4000) #skip every 2 seconds. You need to modify this
TIME_SKIPS_seconds=int(TIME_SKIPS/1000)

day_list = [
    {"label": str(ids), "value": str(ids)}
    for ids in list(range(57))
]

coup_list = [
    {"label": str(ids), "value": str(ids)}
    for ids in list(range(10))
]

boundaries = [
    ([17, 15, 100], [50, 56, 200]),
    ([86, 31, 4], [220, 88, 50]),
    ([25, 146, 190], [62, 174, 250])]
##############################OPENCV

START_FRAME_NUMBER = 0


# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture(VIDEO_PATH)
cap.set(cv.CAP_PROP_POS_FRAMES, START_FRAME_NUMBER)

# Get the first frame of the video
ret, first_frame = cap.read()

# Preprocess the initial frame
prev_gray = cv.GaussianBlur(first_frame,(9,9),1)
prev_gray = cv.cvtColor(prev_gray,cv.COLOR_BGR2GRAY)

# Define background subtractor
fgbg = cv.createBackgroundSubtractorMOG2()

# apply background subtraction mask and morphological opening
prev_gray = fgbg.apply(prev_gray)
kernel = np.ones((3,3),np.uint8)
prev_gray = cv.morphologyEx(prev_gray, cv.MORPH_OPEN, kernel)


# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

# Define the columns to be included in the DataFrame (the motion magnitude will be added to here)
gray_image1 = []
magn_total_image = []
magn_feeding_station = []
magn_drinking_station = []
magn_lamp_area = []
magn_other = []
index = []
count = 0
while(cap.isOpened()):
    cap.set(cv.CAP_PROP_POS_MSEC,(count*TIME_SKIPS))

    ret, frame = cap.read()
    if not ret:
        break
    sumcolor=0
    frame1 = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    for (lower, upper) in boundaries:       
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask1 = cv.inRange(frame1, lower, upper)
        output1 = cv.bitwise_and(frame1, frame1, mask = mask1)
        sumcolor=sumcolor+np.mean(output1)
    if sumcolor==0:
        gray_image1.append(0)
    else:
        gray_image1.append(1)
    # Opens a new window and displays the input frame
    #cv.imshow("input", frame[40:320,200:530])
    

    # Preprocess with the same steps as the first image
    gray = cv.GaussianBlur(frame,(9,9),1)
    gray = cv.cvtColor(gray,cv.COLOR_BGR2GRAY)
    gray = fgbg.apply(gray)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)

    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    # append the magnitudes
    
    magn_total_image.append(sum(sum((magnitude))))
    magn_feeding_station.append(sum(sum(magnitude[185:285,270:365])))
    magn_drinking_station.append(sum(sum(magnitude[75:140,240:530])))
    magn_lamp_area.append(sum(sum(magnitude[140:250,400:530])))
    index.append(count*TIME_SKIPS_seconds)
    count += 1
    
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
      
    # To visualize the processed image: a blend between original and rgb
    blend = cv.addWeighted(frame,0.4, rgb,0.6, 0)

    # Opens a new window and displays the output frame
    #cv.imshow("dense optical flow", blend[40:320,200:530])      
    # Updates previous frame
    prev_gray = gray
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    #if cv.waitKey(50) & 0xFF == ord('q'):
     #   break
        
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()

##############################

##############################CNN
gray_image2 = []
predic_sleeping = []
predic_feeding = []
predic_awake = []
time_sec=[]
count1 = 0

queuesize=128
queue=128
# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
vs = cv2.VideoCapture(VIDEO_PATH)

while True:
    
    vs.set(cv2.CAP_PROP_POS_MSEC,(count1*TIME_SKIPS))    # move the time
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    sumcolor=0
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask1 = cv2.inRange(frame1, lower, upper)
        output1 = cv2.bitwise_and(frame1, frame1, mask = mask1)
        sumcolor=sumcolor+np.mean(output1)
    if sumcolor==0:
        gray_image2.append(0)
    else:
        gray_image2.append(1)
    # if the frame was not grabbed, then we have reached the end
    # of the stream
      
    # clone the output frame, then convert it from BGR to RGB
    # ordering, resize the frame to a fixed 224x224, and then
    # perform mean subtraction
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (300, 300)).astype("float32")
    frame -= mean
    # make predictions on the frame and then update the predictions
    # queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    
    
    predic_awake.append(preds[0])
    predic_feeding.append(preds[1])
    predic_sleeping.append(preds[2])
    time_sec.append(count1*TIME_SKIPS_seconds)
    
    count1 += 1
    

    # draw the activity on the output frame
  

 # release the file pointers
vs.release()
##############################
#check if frame is gray

#############################

# Create global chart template
mapbox_access_token = "pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w"
layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=10, r=10, b=10, t=10),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="North Brabant map",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=5.0919, lat=51.5606), 
        zoom=8,
        width="100%",
       height="100%", 
    ),
)

# Create app layout
app.layout = html.Div(
    [
     dcc.Store(id="aggregate_data"),
     html.Div(id="output-clientside"),
#############################################################  
# first container (logo, title)
     html.Div([ 
           # logo
           html.Div([html.Img(src=app.get_asset_url("logo2.png"),
                              id="plotly-image",
                              style={"height": "80px", "width": "50%", "margin-bottom": "25px",},
                              )    ], 
                     className="one-third column",),
           # main title
           html.Div([html.Div([ html.H3( "Video-based Analytics for Livestock Farming",
                                    style={"margin-bottom": "0px"},),
                                 html.H5("An Overview", style={"margin-top": "0px"}),
                            ])    ],
                     className="one-half column",style={"margin-top": "0px", "margin-right": "29.5%"},
                     id="title",),
         
         
            ],
    id="header",
    className="row flex-display",
    style={"margin-bottom": "25px"},
        ),

################################################################################               
# second container (video, histogram)     
        html.Div([
                           
                html.Div(children = [
                                     
                    
                     html.Div(
                            id="filter-container",style = {'display' : 'flex',"height": "12%"},
                            children=[
                                
                                
                                 html.P(
                            "Coup",
                            className="control_label",
                        ),
                       
                       
                        dcc.Dropdown(
                            id="coup_filter",
                            options=coup_list,
                            multi=False,
                            value='0',
                            className="dcc_control",
                        )                                                      
                                ,
                                
                           html.P(
                            "Day",
                            className="control_label",
                               style = {"margin-left": "15%"}
                                  
                        ),
                       
                       
                        dcc.Dropdown(
                            id="day_filter",
                            options=day_list,
                            multi=False,
                            value='0',
                            className="dcc_control",
                           
                        )
                                ,         
                                         
                               html.Button("Search", id="Search-button", 
                                        style={"margin-top" : "15%",
                                               "margin-left" : "10%", 
                                              }),       
                            
                               
                            ],
                    className="pretty_container",

                        )    
                    
                  ,          
                    
   ###############################        
                  
                    
                     html.Div(
                            id="video-container",
                            children=[
                    
                                
                                 dash_player.DashPlayer( id='video-player',
                                      url=VIDEO_PATH,
                                        controls=True,
                                       # playing=True,
                                        width = 520,
                                        height = 300,),
                                
                                
                
                    
                     ],
                    className="pretty_container",

                        )     , ],
                className="five columns",) ,
            
            
                html.Div( [
                                   
                    dcc.Graph(id="magnitude_graph", 
                                    style={ "width": "100%", 
                                           "position":"relative", "z-index": "2"},
                                   config={'displayModeBar': True}),
                    
                 dcc.Graph(id="CNN_graph", 
                                    style={  "width": "100%", 
                                           "position":"relative", "z-index": "2"},
                                   config={'displayModeBar': True}),
               
                                   ],
                          className="pretty_container seven columns",
                         ),
             ],
            className="row flex-display",
        ),   
     
    
    ],
       id="mainContainer",
       style={"display": "flex", "flex-direction": "column"},
)


@app.callback(Output("magnitude_graph", "figure"),
              [ Input("aggregate_data", "data"), 
              ]
             
             )   
def magnitude_figure(data):
    
    
    points=[]
    firstindex=gray_image1.index(0)
    points.append(index[firstindex])
    
    for i in range(firstindex,len(gray_image1)-1):
        if gray_image1[i+1]==0 and gray_image1[i]==1:
            points.append(index[i])
        if gray_image1[i+1]==1 and gray_image1[i]==0:
            points.append(index[i])     
    
    if (len(points)%2!=0) and gray_image1[-1]==0:
        points.append(index[-1])
    
    traces = []
    
    trace = go.Scatter(
            x=index, 
            y=magn_total_image,
             mode = 'lines',
                   type = 'scatter',
                   name='total',
                   line = dict(shape = 'linear', width= 2),
                    connectgaps = True
        )
    traces.append(trace)
    
    trace = go.Scatter(
            x=index, 
            y=magn_feeding_station,
             mode = 'lines',
                   type = 'scatter',
                   name='feeding area',
                   line = dict(shape = 'linear', width= 2),
                    connectgaps = True
        )
    traces.append(trace)
    
    trace = go.Scatter(
            x=index, 
            y=magn_drinking_station,
             mode = 'lines',
                   type = 'scatter',
                   name='drinking area',
                   line = dict(shape = 'linear', width= 2),
                    connectgaps = True
        )
    traces.append(trace)
    
    
    
    allshapes=[]
    inde=0
    while inde<=(len(points)-2):
        allshapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=points[inde],
            y0=0,
            x1=points[inde+1],
            y1=1,
            fillcolor="#d3d3d3",
            opacity=0.5,
            layer="below",
            line_width=0,
        ) )
        inde=inde+2
       
    layout=go.Layout(title_text ='Where does activity take place?',xaxis = dict(title = 'Minutes'),
                     yaxis = dict(title = 'Motion Magnitude'),shapes=allshapes)  
    
   
    
    figure = dict(data = traces, layout = layout)
    
    
    return figure



@app.callback(Output("CNN_graph", "figure"),
              [ Input("aggregate_data", "data")
               
              ]
             
             )   
def cnn_figure(data):
    
  
    
    points=[]
    firstindex=gray_image2.index(0)
    points.append(time_sec[firstindex])
    
    for i in range(firstindex,len(gray_image2)-1):
        if gray_image2[i+1]==0 and gray_image2[i]==1:
            points.append(time_sec[i])
        if gray_image2[i+1]==1 and gray_image2[i]==0:
            points.append(time_sec[i])     
    
    if (len(points)%2!=0) and gray_image2[-1]==0:
        points.append(time_sec[-1])
    
    traces = []
    
    trace = go.Scatter(
            x=time_sec, 
            y=predic_sleeping,
             mode = 'lines',
                   type = 'scatter',
                   name='sleeping',
                   line = dict(shape = 'linear', width= 2),
                    connectgaps = True
        )
    traces.append(trace)
    
    trace = go.Scatter(
            x=time_sec, 
            y=predic_feeding,
             mode = 'lines',
                   type = 'scatter',
                   name='feeding',
                   line = dict(shape = 'linear', width= 2),
                    connectgaps = True
        )
    traces.append(trace)
    
    trace = go.Scatter(
            x=time_sec, 
            y=predic_awake,
             mode = 'lines',
                   type = 'scatter',
                   name='awake',
                   line = dict(shape = 'linear', width= 2),
                    connectgaps = True
        )
    traces.append(trace)
    
        
    allshapes=[]
    inde=0
    while inde<=(len(points)-2):
        allshapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=points[inde],
            y0=0,
            x1=points[inde+1],
            y1=1,
            fillcolor="#d3d3d3",
            opacity=0.5,
            layer="below",
            line_width=0,
        ) )
        inde=inde+2
    
    layout=go.Layout(title_text ='prediction',xaxis = dict(title = 'Minutes'),
                     yaxis = dict(title = 'Activity Prediction'),
                     shapes=allshapes)   
    
   
    
    
    figure = dict(data = traces, layout = layout)
    
  
    
    return figure

@app.callback(Output('video-player', 'seekTo'),
              [Input("magnitude_graph", "relayoutData"),
              Input("CNN_graph", "relayoutData")])
def set_seekTo(magdata,cnndata):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if  (magdata) and ('magnitude_graph.relayoutData' in changed_id) and ('xaxis.range[0]' in magdata.keys()):
        return magdata['xaxis.range[0]']
    elif (cnndata) and ('CNN_graph.relayoutData' in changed_id) and ('xaxis.range[0]' in cnndata.keys()):
        return cnndata['xaxis.range[0]']
    

# Main
if __name__ == "__main__":
    app.run_server(debug=False)


# In[ ]:




