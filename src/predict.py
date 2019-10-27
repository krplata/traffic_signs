from keras.models import load_model
from keras.utils import plot_model

model = load_model('models/color.h5')
plot_model(model, to_file='output.png', show_shapes=True)

