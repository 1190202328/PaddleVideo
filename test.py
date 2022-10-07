import paddle

paddle.utils.run_check()

from ppvideo import PaddleVideo

clas = PaddleVideo(model_name='ppTSM', use_gpu=True)
video_file = 'data/example.avi'
clas.predict(video_file)
