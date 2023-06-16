import polars as pl
df = pl.read_parquet('../data/train_annoatations.parquet')

print(df['chart-type'].value_counts())

classes = ["vertical_bar",
"horizontal_bar",
"line",
"dot", "scatter"]

folder = [ "/data/images/graphs_v/",
"/data/images/graphs_h/",
"/data/images/graphs_l/",
"/data/images/graphs_d/",
"/data/images/graphs_s/"
]

main = df.with_columns(
    pl.col("chart-type").apply(lambda x: classes.index(x)),
    pl.col('id')
)[['chart-type', 'id']]

image_processed_folder = '/workspace/Benetech-Kaggle-Competition'
image_folder = '/workspace/data/benetech/train/images/'

import os
import shutil
# take values in each row
for i in range(len(main)):
    chart_type = main['chart-type'][i]
    image_id = main['id'][i]
    image_path = image_folder + image_id + '.jpg'
    image_processed_path = image_processed_folder + folder[chart_type] + image_id + '.jpg'
    if os.path.exists(image_path) and not os.path.exists(image_processed_path):
        shutil.copy(image_path, image_processed_path)
    else:
        print(image_path)

