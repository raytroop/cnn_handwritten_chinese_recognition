import base64
import pickle
from flask import Flask
from flask import render_template, json, jsonify, request
from app.utils import predictPrepare, createImage, imagePrepare
from .consts import __test_image_file, __pred1_image_file, __pred2_image_file, __pred3_image_file, \
    __code_to_chinese_file

app = Flask(__name__)

global_graph, global_session = None, None


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global global_graph, global_session
    global_graph, global_session = predictPrepare()  # 加载模型，准备好预测


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title='Home')


@app.route('/chineseRecognize', methods=['POST'])
def chineseRecognize():
    # 接受前端发来的数据
    data = json.loads(request.form.get('data'))
    imagedata = data["test_image"]
    imagedata = imagedata[22:]
    img = base64.b64decode(imagedata)
    file = open(__test_image_file, 'wb')
    file.write(img)
    file.close()

    temp_image = imagePrepare(__test_image_file)
    predict_val, predict_index = global_session.run(
        [global_graph['predicted_val_top_k'], global_graph['predicted_index_top_k']],
        feed_dict={global_graph['images']: temp_image, global_graph['keep_prob']: 1.0})
    with open(__code_to_chinese_file, 'rb') as f2:
        word_dict = pickle.load(f2)
    createImage(word_dict[predict_index[0][0]], __pred1_image_file)
    createImage(word_dict[predict_index[0][1]], __pred2_image_file)
    createImage(word_dict[predict_index[0][2]], __pred3_image_file)

    # 将识别图片转码传给前端，并带上对应的准确率
    with open(__pred1_image_file, 'rb') as fin:
        image1_data = fin.read()
        pred1_image = base64.b64encode(image1_data)
    with open(__pred2_image_file, 'rb') as fin:
        image2_data = fin.read()
        pred2_image = base64.b64encode(image2_data)
    with open(__pred3_image_file, 'rb') as fin:
        image3_data = fin.read()
        pred3_image = base64.b64encode(image3_data)
    info = dict()
    info['pred1_image'] = "data:image/jpg;base64," + pred1_image.decode()
    info['pred1_accuracy'] = str('{:.2%}'.format(predict_val[0][0]))
    info['pred2_image'] = "data:image/jpg;base64," + pred2_image.decode()
    info['pred2_accuracy'] = str('{:.2%}'.format(predict_val[0][1]))
    info['pred3_image'] = "data:image/jpg;base64," + pred3_image.decode()
    info['pred3_accuracy'] = str('{:.2%}'.format(predict_val[0][2]))
    return jsonify(info)


def launch_server():
    print(("* Loading TensorFlow model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(debug=True, host='127.0.0.1', port=5000)
