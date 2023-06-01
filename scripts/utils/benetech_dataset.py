import json 
import os
from PIL import Image
import plotly.graph_objects as go
import polars as pl
from typing import Dict, List, Tuple, Union

class BenetechDataset:
    def __init__(self, CFG):
        self.CFG = CFG
        self.annotations = self.get_annotations()
        
    def get_annotations(self):
        self.annotations = []
        for annotation_path in os.listdir(self.CFG.train_ann_path):
            with open(f"{self.CFG.train_ann_path}/{annotation_path}") as annotation_f:
                dct = json.load(annotation_f)
                dct['id'] = annotation_path.split('.')[0]
                self.annotations.append(dct)
        return self.annotations

    def to_pl_df(self):
        def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
            """Flatten a nested dictionary"""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        data = []
        for dataaaa in self.annotations:
            # Flatten the nested dictionaries
            flat_data = flatten_dict(dataaaa)
            data.append(flat_data)
        df = pl.DataFrame(data)
        return df


    def get_chart_type_counts(self):
        chart_type_counts = {
            "dot": 0,
            "line": 0,
            "scatter": 0,
            "vertical_bar": 0,
            "horizontal_bar": 0
        }
        
        for annotation in self.annotations:
            chart_type_counts[annotation["chart-type"]] += 1
                
        return chart_type_counts

    def load_annotation(self, name: str):
        with open(f"{self.CFG.train_ann_path}/{name}.json") as annotation_f:
            ann_example = json.load(annotation_f)
        return ann_example

    def get_coords(self, polygon, img_height):
        xs = [
            polygon["x0"], 
            polygon["x1"], 
            polygon["x2"], 
            polygon["x3"], 
            polygon["x0"]
        ]
        
        ys = [
            -polygon["y0"] + img_height, 
            -polygon["y1"] + img_height, 
            -polygon["y2"] + img_height, 
            -polygon["y3"] + img_height, 
            -polygon["y0"] + img_height
        ]
        
        return xs, ys


    def add_line_breaks(self, text: str, break_num: int = 7) -> str:
        words = text.split()
        new_text = ""
        for i, word in enumerate(words, start=1):
            new_text += word
            if i % break_num == 0:
                new_text += "<br>"
            else:
                new_text += " "
        return new_text

    def get_tick_value(self, name, data_series):
        for el in data_series:
            if el["x"] == name:
                return el["y"]
            elif el["y"] == name:
                return el["x"]

    def plot_annotated_image(self, name: str, scale_factor: int = 1.0) -> None:
        img_example = Image.open(f"{self.CFG.train_img_path}/{name}.jpg")
        ann_example = self.load_annotation(name)
        
        # create figure
        fig = go.Figure()

        # constants
        img_width = img_example.size[0]
        img_height = img_example.size[1]
        

        # add invisible scatter trace
        fig.add_trace(
            go.Scatter(
                x=[0, img_width],
                y=[0, img_height],
                mode="markers",
                marker_opacity=0
            )
        )

        # configure axes
        fig.update_xaxes(
            visible=False,
            range=[0, img_width]
        )

        fig.update_yaxes(
            visible=False,
            range=[0, img_height],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x"
        )

        # add image
        fig.add_layout_image(dict(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x", yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img_example
        ))
        
        # add bounding box
        fig.add_shape(
            type="rect",
            x0=ann_example["plot-bb"]["x0"], 
            y0=-ann_example["plot-bb"]["y0"] + img_height, 
            x1=ann_example["plot-bb"]["x0"] + ann_example["plot-bb"]["width"], 
            y1=-(ann_example["plot-bb"]["y0"] + ann_example["plot-bb"]["height"]) + img_height,
            line=dict(color="RoyalBlue"),
        )
        
        # add polygons
        for text in ann_example["text"]:
            name = text["text"]
            
            if text["role"] == "tick_label":
                tick_value = self.get_tick_value(name, ann_example["data-series"])
                if tick_value:
                    name = f'Text: {name}<br>Value: {tick_value}'
            
            xs, ys = self.get_coords(text["polygon"], img_height)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, fill="toself",
                name=self.add_line_breaks(name),
                hovertemplate="%{name}",
                mode='lines'
            ))
        
        # add x-axis dots
        xs = [dot["tick_pt"]["x"] for dot in ann_example["axes"]["x-axis"]["ticks"]]
        ys = [-dot["tick_pt"]["y"] + img_height for dot in ann_example["axes"]["x-axis"]["ticks"]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers',
            name="x-axis"
        ))
        
        # add y-axis dots
        xs = [dot["tick_pt"]["x"] for dot in ann_example["axes"]["y-axis"]["ticks"]]
        ys = [-dot["tick_pt"]["y"] + img_height for dot in ann_example["axes"]["y-axis"]["ticks"]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers',
            name="y-axis"
        ))

        # configure other layout
        fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            showlegend=False
        )

        # disable the autosize on double click because it adds unwanted margins around the image
        # and finally show figure
        fig.show(config={'doubleClick': 'reset'})
