import gradio as gr
import numpy as np
import cv2
import supervision as sv    # For annotations
from ultralytics import YOLO
import glob
import json
import ast

# TODO: finetune/test bigger models
model_1 = YOLO('best.pt')   # Finetuned YoloV8s 
# model_2 = 
# model_3 =

box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

def show_preds_image(option, image_path):

    predict = []

    if(option == "yolov8s-ft-yalta-ai-segmonto-manuscript"):

        model = model_1

    # elif(option == "yolov8m-ft-yalta-ai-segmonto-manuscript"):
    #     model = model_2
    # else(option == "yolov8l-ft-yalta-ai-segmonto-manuscript"):
    #     model = model_3
    
    image = cv2.imread(image_path)

    outputs = model.predict(source=image_path, device="cpu")
 
 ##############
    # result = outputs[0]
    # bboxes = np.array(result.boxes.xyxy, dtype="int") # result.boxes.xyxy.cpu()
    # classes = np.array(result.boxes.cls, dtype="int")
    
    # for cls, bbox in zip(classes, bboxes):
    #     (x, y, x2, y2) = bbox
    #     cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 3)
    #     # cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
    #     cv2.putText(frame, str(model.names[int(cls)]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
################

    result = outputs[0]
    # detections = sv.Detections.from_yolov8(result)    # Deprecated
    detections = sv.Detections.from_ultralytics(result)

    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]

    frame = box_annotator.annotate(
        scene=image, 
        detections=detections, 
        labels=labels
    )

    # Build the dictionary
    predict.append(
        {
            "label": [ast.literal_eval(model.model.names[id]) for id in detections.class_id.tolist()],
            # The list of coordinates of the points of the polygon.
            "bbox": detections.xyxy.tolist(),
            # Confidence that the model predicts the polygon in the right place
            "confidence": detections.confidence.tolist(), 
        }
    )

    # captions = {
    #     f"{model.model.names[class_id]}": float("{:.2f}".format(confidence))
    #     for _, _, confidence, class_id, _
    #     in detections
    # }

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), json.dumps(predict, indent=2)#, captions
 
title = "<h1 style='text-align: center'>YoloV8 Medieval Manuscript Region Detection 📜🪶 - SegmOnto Ontology</h1>"
description="""Treating page layout recognition on historical documents as an object detection task (compared to the usual pixel segmentation approach). Model finetuned on **YALTAi Segmonto Manuscript and Early Printed Book Dataset** (HF `dataset card`: [biglam/yalta_ai_segmonto_manuscript_dataset](https://huggingface.co/datasets/biglam/yalta_ai_segmonto_manuscript_dataset)).
* Note that this demo is running on a small resource environment, `basic CPU plan` (`2 vCPU, 16GB RAM`).
"""

article = "<p style='text-align: center'>ArXiv: <a href='https://arxiv.org/abs/2207.11230v1' target='_blank'>You Actually Look Twice At it (YALTAi): using an object detection approach instead of region segmentation within the Kraken engine</a></p>"

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.HTML(title)
    gr.Markdown(description)
    # gr.HTML(description)

    with gr.Row():

        with gr.Column(scale=1, variant="panel"):

            with gr.Row():

                input_image = gr.components.Image(type="filepath", label="Input Image", height=350)
            
            with gr.Row():

                input_model = gr.components.Dropdown(["yolov8s-ft-yalta-ai-segmonto-manuscript"], label="Model")
            
            with gr.Row():

                btn_clear = gr.Button(value="Clear")
                btn = gr.Button(value="Submit")
            
            with gr.Row():

                with gr.Accordion(label="Choose an example:", open=False):

                    gr.Examples(
                        examples = [["yolov8s-ft-yalta-ai-segmonto-manuscript", str(file)] for file in glob.glob("./examples/*.jpg")],
                        inputs = [input_model, input_image],
                        # label="Samples",
                    )
    
        with gr.Column(scale=1, variant="panel"):

            with gr.Tab("Output"):

                with gr.Row():

                    output = gr.components.Image(type="numpy", label="Output", height=500)
                
                # with gr.Row():
                #     btn_flag = gr.Button(value="Flag")  # TODO

                # with gr.Row():
                #     captions = gr.Dataframe(headers=["Label", "Confidence"])
            
            with gr.Tab("JSON Output"):

                with gr.Row():

                    with gr.Column():

                        with gr.Accordion(label="JSON Output", open="False"):

                            # Generates a json with the model predictions
                            json_output = gr.JSON(label="JSON")

        btn.click(show_preds_image, inputs=[input_model, input_image], outputs=[output, json_output])
        btn_clear.click(lambda: [None, None, None, None], outputs=[input_image, input_model, output, json_output])
        # btn_flag.click()
    
    with gr.Row():

        gr.HTML(article)    

if __name__ =="__main__":

    demo.queue().launch()   # share=True, auth=("username", "password")