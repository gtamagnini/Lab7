#pip install opencv-python
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Importa e converte para RGB
img = cv2.imread('Aviao.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Filtro de ruído (bluring)
img_blur = cv2.blur(img, (5, 5))

# Convertendo para preto e branco (RGB -> Gray Scale -> BW)
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, a / 2 + 100, a, cv2.THRESH_BINARY_INV)

# Preparando o "kernel"
kernel = np.ones((12, 12), np.uint8)

# Operadores Morfológicos
img_dilate = cv2.dilate(thresh, kernel, iterations=1)
img_erode = cv2.erode(thresh, kernel, iterations=1)
img_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
img_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
img_grad = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
img_tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)
img_blackhat = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel)

# Plot the images
imagens = [img, img_blur, img_gray, thresh, img_erode, img_dilate, img_open, img_close, img_grad, img_tophat, img_blackhat]

# Ajusta o tamanho da figura
plt.figure(figsize=(12, 10))  # Ajuste conforme necessário

# Cria um GridSpec para o layout
gs = gridspec.GridSpec(4, 3)  # 4 linhas e 3 colunas

# Adiciona as imagens ao GridSpec
for i in range(len(imagens) - 1):  # Para todas as imagens, exceto a última
    ax = plt.subplot(gs[i // 3, i % 3])
    ax.imshow(imagens[i], 'gray')
    ax.axis('off')

# Para a última imagem, ocupando mais espaço
ax_last = plt.subplot(gs[2:, :])  # A última imagem ocupa as duas últimas linhas
ax_last.imshow(img_blackhat, 'gray')  # Mude para a imagem que você deseja exibir como maior
ax_last.axis('off')

# Exibe a imagem
plt.tight_layout()
plt.show()
