from utils import Lane_Finder
import matplotlib.pyplot as plt
import cv2

test_img = cv2.imread('test_images/hard_test4.jpg')
lane_finder = Lane_Finder()

bin = lane_finder.threshold_binary(test_img)
bin_warp = cv2.warpPerspective(bin, lane_finder.warp_transform, lane_finder.resolution, flags=cv2.INTER_LINEAR)
hls = cv2.cvtColor(test_img, cv2.COLOR_BGR2HLS)

res_img = lane_finder.pipeline(test_img)
res_img = lane_finder.pipeline(test_img)
res_img = lane_finder.pipeline(test_img)
res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

f, axes = plt.subplots(4, 1, figsize=(6, 10))
axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[1].imshow(hls)
axes[2].imshow(bin, cmap='gray', vmin=0, vmax=1)
axes[3].imshow(bin_warp, cmap='gray', vmin=0, vmax=1)
#axes[3].imshow(res_img)
plt.show()
