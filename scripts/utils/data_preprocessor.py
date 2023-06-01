import torch
from typing import List
import re
import 

class DonutDataProcessor:
    def __init__(self, cfg):
        """
        Initializes the DonutDataProcessor object.

        Parameters
        ----------
        cfg : object
            Configuration object.
        """
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        self.model = None
        self.processor = None
        self.decoder_start_token_id = None
        self.ds = None
        self.data_loader = None

    @staticmethod
    def clean(str_list: List[str]):
        """
        Cleans a list of stringified numbers by removing unwanted characters and 
        converting to numerical format.

        Parameters
        ----------
        str_list : list of str
            The list of stringified numbers to clean.

        Returns
        -------
        list of float
            A new list where each entry is the cleaned numerical version of 
            the corresponding entry in the input list.
        """
        new_list = []
        for temp in str_list:
            if "." not in temp:
                dtype = int
            else:
                dtype = float
            try:
                temp = dtype(re.sub("\s", "", temp))
            except ValueError:
                temp = re.sub(r"[^0-9\.\-eE]", "", temp)

                if len(temp) == 0:
                    temp = 0
                else:
                    multiple_periods = len(re.findall(r"\.", temp)) > 1
                    multiple_negative_signs = len(re.findall(r"\-", temp)) > 1
                    multiple_e = len(re.findall(r"[eE]", temp)) > 1

                    if multiple_negative_signs:
                        temp = "-" + re.sub(r"\-", "", temp)

                    if multiple_periods:
                        chunks = temp.split(".")
                        try:
                            temp = chunks[0] + "." + "".join(chunks[1:])
                        except IndexError:
                            temp = "".join(chunks)

                    if multiple_e:
                        while temp.lower().startswith("e"):
                            temp = temp[1:]

                        while temp.lower().endswith("e"):
                            temp = temp[:-1]

                        chunks = temp.split("e")
                        try:
                            temp = chunks[0:-1] + "e" + "".join(chunks[-1])
                        except IndexError:
                            temp = "".join(chunks)
                try:
                    temp = dtype(temp)
                except ValueError:
                    temp = 0

            new_list.append(temp)

        return new_list

    def clean_preds(self, x: List[str], y: List[str]):
        """
        This function cleans the x and y values predicted by Donut.

        Because it is a generative model, it can insert any character in the
        model's vocabulary into the prediction string. This function primarily removes
        characters that prevent a number from being cast to a float.

        Parameters
        ----------
        x : list of str
            The x values predicted by Donut.
        y : list of str
            The y values predicted by Donut.

        Returns
        -------
        list of str, list of str
            The cleaned x values and cleaned y values.
        """
        all_x_chars = "".join(x)
        all_y_chars = "".join(y)

        frac_num_x = len(re.sub(r"[^\d]", "", all_x_chars)) / len(all_x_chars)
        frac_num_y = len(re.sub(r"[^\d]", "", all_y_chars)) / len(all_y_chars)

        if frac_num_x >= 0.5:
            x = self.clean(x)
        else:
            x = [s.strip() for s in x]

        if frac_num_y >= 0.5:
            y = self.clean(y)
        else:
            y = [s.strip() for s in y]

        return x, y

    def string2preds(self, pred_string: str):
        """
        Convert the prediction string from Donut to a chart type and x and y values.

        Checks to make sure the special tokens are present and that the x and y values are not empty.
        Will truncate the list of values to the smaller length of the two lists. This is because the
        lengths of the x and y values must be the same to earn any points.

        Parameters
        ----------
        pred_string : str
            The prediction string from Donut.

        Returns
        -------
        str, list of str, list of str
            The chart type predicted by Donut, the x values, and the y values.
        """
        if "<dot>" in pred_string:
            chart_type = "dot"
        elif "<horizontal_bar>" in pred_string:
            chart_type = "horizontal_bar"
        elif "<vertical_bar>" in pred_string:
            chart_type = "vertical_bar"
        elif "<scatter>" in pred_string:
            chart_type = "scatter"
        elif "<line>" in pred_string:
            chart_type = "line"
        else:
            return "vertical_bar", [], []

        if not all([x in pred_string for x in [self.cfg.X_START, 
                                               self.cfg.X_END, 
                                               self.cfg.Y_START, 
                                               self.cfg.Y_END]]):
            return chart_type, [], []

        pred_string = re.sub(r"<one>", "1", pred_string)

        x = pred_string.split(self.cfg.X_START)[1].split(self.cfg.X_END)[0].split(";")
        y = pred_string.split(self.cfg.Y_START)[1].split(self.cfg.Y_END)[0].split(";")

        if len(x) == 0 or len(y) == 0:
            return chart_type, [], []

        x, y = self.clean_preds(x, y)

        return chart_type, x, y

    def preprocess(self, examples, processor):
        """
        Preprocess the images using the defined processor.

        This method converts the input sample (image) into an array and preprocesses it using the processor.
        If the image is grayscale, it is converted into a 3-channel format.

        Parameters
        ----------
        examples : dict
            Dictionary containing input examples.
        processor : DonutProcessor
            The DonutProcessor object for preprocessing.

        Returns
        -------
        dict
            A dictionary containing pixel values of the preprocessed image.
        """
        pixel_values = []

        for sample in examples["image_path"]:
            arr = np.array(sample)
            if len(arr.shape) == 2:
                arr = np.stack([arr]*3, axis=-1)
        
            pixel_values.append(processor(arr, random_padding=True).pixel_values)

        return {"pixel_values": torch.tensor(np.vstack(pixel_values))}

    def __load_model(self):
        """Load the pre-trained model and set it to evaluation mode."""
        self.model = VisionEncoderDecoderModel.from_pretrained(self.cfg.model_donut_dir)
        self.model.eval()
        self.model.to(self.device)
        self.decoder_start_token_id = self.model.config.decoder_start_token_id
        self.processor = DonutProcessor.from_pretrained(self.cfg.model_donut_dir)

    def load_dataset(self):
        """
        Load the dataset and create a data loader for batching.

        Returns
        -------
        list of str
            List of image IDs.
        """
        self.__load_model()
        image_dir = Path(self.cfg.image_path)
        images = list(image_dir.glob("*.jpg"))
        self.ds = Dataset.from_dict(
            {"image_path": [str(x) for x in images], "id": [x.stem for x in images]}
        ).cast_column("image_path", ds_img())
        ids = self.ds["id"]
        self.ds.set_transform(partial(self.preprocess, processor=self.processor))
        self.data_loader = DataLoader(
            self.ds, batch_size=self.cfg.batch_size, shuffle=False
        )
        return ids

    def generate_predictions(self):
        """
        Generate predictions for the input images.

        Returns
        -------
        list of str, list of str, list of str
            List of chart types, x predictions, and y predictions.
        """
        all_generations = []
        for batch in tqdm(self.data_loader):
            pixel_values = batch["pixel_values"].to(self.device)
            batch_size = pixel_values.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.decoder_start_token_id,
                device=pixel_values.device,
            )

            try:
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.cfg.max_length,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=4,
                    temperature=.7,
                    top_k=3,
                    top_p=.5,
                    return_dict_in_generate=True,
                )

                all_generations.extend(self.processor.batch_decode(outputs.sequences))

            except:
                all_generations.extend([""] * batch_size)

        chart_types, x_preds, y_preds = [], [], []
        for gen in all_generations:
            try:
                chart_type, x, y = self.string2preds(gen)
                new_chart_type = chart_type
                x_str = ";".join(list(map(str, x)))
                y_str = ";".join(list(map(str, y)))

            except Exception as e:
                new_chart_type = self.cfg.PLACEHOLDER_CHART_TYPE
                x_str = self.cfg.PLACEHOLDER_DATA_SERIES
                y_str = self.cfg.PLACEHOLDER_DATA_SERIES

            if len(x_str) == 0:
                x_str = self.cfg.PLACEHOLDER_DATA_SERIES
            if len(y_str) == 0:
                y_str = self.cfg.PLACEHOLDER_DATA_SERIES

            chart_types.append(new_chart_type)
            x_preds.append(x_str)
            y_preds.append(y_str)

        return chart_types, x_preds, y_preds