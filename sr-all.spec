# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a_sr_train = Analysis(['sr-train.py'],
             binaries=[], 
             pathex=[], datas=[], hiddenimports=[], runtime_hooks=[],
             #hookspath=['myhooks'],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter', 'PyQt5', 'matplotlib', 'PIL',  'pywt', 'google-cloud-core', 'win32com', 'zmq'],
             win_no_prefer_redirects=False, win_private_assemblies=False, noarchive=False,
             cipher=block_cipher
             )
             
a_sr_image = Analysis(['sr-image.py'],
             binaries=[], 
             pathex=[], datas=[], hiddenimports=[], runtime_hooks=[],
             #hookspath=['myhooks'],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter', 'PyQt5', 'matplotlib', 'PIL',  'pywt', 'google-cloud-core', 'win32com', 'zmq'],
             win_no_prefer_redirects=False, win_private_assemblies=False, noarchive=False,
             cipher=block_cipher
             )
             
a_sr_video = Analysis(['sr-video.py'],
             binaries=[('ffmpeg/bin/*', '.')],
             pathex=[], datas=[], hiddenimports=[], runtime_hooks=[],
             #hookspath=['myhooks'],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter', 'PyQt5', 'matplotlib', 'PIL',  'pywt', 'google-cloud-core', 'win32com', 'zmq'],
             win_no_prefer_redirects=False, win_private_assemblies=False, noarchive=False,
             cipher=block_cipher
             )
             
pyz_train = PYZ(a_sr_train.pure, 
                a_sr_train.zipped_data,
                cipher=block_cipher)
pyz_image = PYZ(a_sr_image.pure, 
                a_sr_image.zipped_data,
                cipher=block_cipher)
pyz_video = PYZ(a_sr_video.pure, 
                a_sr_video.zipped_data,
                cipher=block_cipher)

exe_train = EXE(pyz_train,
          a_sr_train.scripts,
          [],
          name='sr-train',
          exclude_binaries=True, debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True
          )

exe_image = EXE(pyz_image,
          a_sr_image.scripts,
          [],
          name='sr-image',
          exclude_binaries=True, debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True
          )
          
exe_video = EXE(pyz_video,
          a_sr_video.scripts,
          [],
          name='sr-video',
          exclude_binaries=True, debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True
          )


coll = COLLECT(exe_train,
               exe_image,
               exe_video,
               a_sr_train.binaries,
               a_sr_train.zipfiles,
               a_sr_train.datas,
               a_sr_image.binaries,
               a_sr_image.zipfiles,
               a_sr_image.datas,
               a_sr_video.binaries,
               a_sr_video.zipfiles,
               a_sr_video.datas,
               strip=False, upx=True,
               name='sr-all')
