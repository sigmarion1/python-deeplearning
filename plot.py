import function as f
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

img = imread('lenna.png')

plt.subplot(3, 1, 1)
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin&cos')
plt.legend()
plt.imshow(img)


x = np.arange(-5.0, 5.0, 0.1)
y = f.step_function(x)

plt.subplot(3, 1, 2)
plt.plot(x, y)

x = np.arange(-5.0, 5.0, 0.1)
y = f.sigmoid(x)

plt.subplot(3, 1, 3)
plt.plot(x, y)

plt.show()
