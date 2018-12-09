from PIL import Image
import glob
import os
import errno

def main():
	img_names = sorted(glob.glob('*.png'), key=os.path.getsize, reverse=True)
	try:
		os.makedirs('cropped')
	except OSError as err:
		if err.errno != errno.EEXIST:
			raise
	for idx, name in enumerate(img_names):
		print('%02d. %s' % (idx, name))
		img = Image.open(name)
		base, ext = os.path.splitext(os.path.basename(name))
		fw, fh = img.size
		hw, hh = fw / 2, fh / 2
		for x0, x1, xn in ((0, hw, 'l'), (hw, fw, 'r')):
			for y0, y1, yn in ((0, hh, 't'), (hh, fh, 'b')):
				crop = img.crop((x0, y0, x1, y1))
				crop.save(os.path.join('cropped', '%s-%s%s%s' % (base, xn, yn, ext)))


if __name__ == '__main__':
	main()