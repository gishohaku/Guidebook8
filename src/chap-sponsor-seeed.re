= NVIDIA Jetson Nanoの消費電力

//flushright{
Seeed 松岡
//}

AI界隈、活況ですね。MidjourneyやStable Diffusionなどの画像生成で賑わっていたかと思うと、今度はAIチャットサービスChatGPTの話でもちきりです。ワクワクしますね。ハードウェアを設計製造している弊社SeeedでもAI関連のハードウェアをいくつか提供しています。最近はNVIDIA Jetsonシリーズのキャリアボードやボックス型コンピュータのラインナップがもりもり増えています。

NVIDIA Jetsonと聞くと、Jetson Nano Developer Kit(@<img>{image1})を思い浮かべる方が多いのではないでしょうか。2019年に販売開始したNVIDIA製GPUを備えた組み込みシステム向けシングルボードコンピュータで、NVIDIAプラットフォームで作られたAIソフトウェアを動かすことができます。$100を切る低価格で日本国内での入手性も良いことから、多くの方々が手に取って動かし楽しんでいると思います。

//image[image1][Jetson Nano Demeloper Kit]

このNVIDIA Jetson、特徴の1つに消費電力があります。一般的なGPUは並列実行と大量な電力で膨大な量の計算を速く処理することを目指していますが、大量な電力を必要とする点は組み込みシステムに少し不向きです。自走式ロボットなど、電源がバッテリーで供給されていて、使える電力が制限されているときがあるからです。NVIDIA JetsonはWebページ（https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/）に「Jetson is compatible with the same AI software and cloud-native workflows used across other NVIDIA platforms and delivers the power-efficient performance customers need to build software-defined autonomous machines」とあるように、電力効率を重視して作られています。NVIDIA Jetsonモジュールの仕様を見ると、消費電力(@<img>{image2})はだいたい5~60Wです。そこで今回はNVIDIA Jetsonモジュールの中で最も消費電力の少ないJetson Nanoモジュールの消費電力を、弊社製品のreComputer J1020で確認してみようと思います。

//image[image2][NVIDIA Jetsonモジュールト消費電力]

== Jetson Nanoモジュールの消費電力を見てみよう

reComputer J1020（https://www.seeedstudio.com/Jetson-10-1-H0-p-5335.html）は製品版Jetson Nanoモジュールをキャリアボードに載せてアルミケースに入れた製品です(@<img>{image3})。あえて“製品版”と書いているのはJetson Nano Developer KitやJetson Nano 2GB Developer Kitとは違うからです（Developer Kitに載っているJetson NanoモジュールはeMMCが無かったり、メモリが少なかったりします）。reComputer J1020を動かすには追加でディスプレイとキーボード、マウスが必要です。また、付属しているACアダプタはPSE認証が無いので使ってはいけません。9~19VのACアダプタ（2.1mm/5.5mmプラグ、センタープラス）も用意しなければいけません。

//image[image3][reComputer J1020]

https://www.seeedstudio.com/Jetson-10-1-H0-p-5335.html

消費電力は、reComputer J1020にUSBカメラを付けてYOLOv7を使って物体検出（Object Detection）しているときに測ることにします(@<img>{image4})。機材は@<img>{image5)のとおりです。reComputer J1020にACアダプタ、USBカメラ、ディスプレイ、キーボード、マウスをつなぎます。ストレージにJetson Nanoモジュールにある16GB eMMCを使いますが、YOLOv7を動かすには容量が足りないので256GB NVMe SSDを追加します。Jetson Nanoモジュールの放熱がやや不安なので、ヒートシンクを弊社製品のアクティブヒートシンクに交換しておきます。Jetson Nanoモジュールが高温になるとヒートシンクに付いているファンが回って強制冷却してくれるので、少し安心です。

//image[image4][実験の様子]

//image[image5][実験に使う機材]

次はソフトウェアです。ブログ（https://lab.seeed.co.jp/entry/2022/06/06/120000）を参考に、Jetson LinuxとJetson stats、JetPackをeMMCへインストールします。執筆時点のJetson Linux最新バージョンは32.7.3でした。そして、NVMe SSDをフォーマットして下記スクリプトで実験に必要なパッケージなどをインストール、セットアップします。

//emlist{
# 使用するパッケージをインストール
$ sudo apt update && sudo apt upgrade
$ sudo apt install python3.8 python3.8-venv curl

# NVMe SSDをマウントしてカレントディレクトリを変更
$ sudo mkdir /mnt/ssd
$ sudo mount /dev/nvme0n1p1 /mnt/ssd
$ cd /mnt/ssd

# 実験用スクリプトをクローン
$ git clone https://github.com/SeeedJP/gishohaku8
$ cd gishohaku8

# Python 3.8 環境を作成してセットアップ
$ python3.8 -m venv venv
$ source venv/bin/activate
$ python -m pip install --upgrade pip
$ wget https://nvidia.box.com/shared/static/2sv2fv1wseihaw8ym0d4srz41dzljwxh.whl
 -O onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl
$ python -m pip install onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl
$ rm onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl
$ python -m pip install opencv-python

# モデルファイルをダウンロード
$ cd models
$ wget https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/main/307_YOLOv7/download
_single_batch.sh
$ chmod +x ./download_single_batch.sh
$./download_single_batch.sh
$ cd ..
//}

準備が整ったので、物体検出を実行しましょう。Python仮想環境venvを利用しているので、gishohaku8ディレクトリでアクティブにしてからobject_detection.pyを起動します。新しいウィンドウが開いてカメラの映像と物体検出のバウンディングボックスが表示されれば正常です。

//emlist{
# 初回のみ
$ cd /mnt/ssd/gishohaku8
$ source venv/bin/activate
$ export DISPLAY=:0.0

$ python ./object_detection.py –provider cuda     # GPUで計算
$ python ./object_detection.py –provider cpu      # CPUで計算
//}

消費電力はJetson Nanoモジュールに内蔵のセンサ(INA3221)で測ります。tegrastatsコマンドやjtopコマンドで表示される電力も、このセンサの値です。jpower.shを起動するとタイムスタンプと消費電力を表示します。消費電力の表示のPOM_5V_IN、POM_5V_GPU、POM_5V_CPUは、モジュール全体の消費電力、GPUの消費電力、CPUの消費電力をミリワットを示しています。

//emlist{
$ sudo /mnt/ssd/gishohaku8/jpower.sh
//}

パワーモード（後記）をMAXNと5WにしてGPUとCPUで物体検出しているときの消費電力をグラフにしたのが@<img>{image6}です。GPUはobject_detection.pyを起動してから推論(Inference)が始まるまで10～15秒ほど時間がかかりました。また、初回の推論時間が遅く(4～6秒)、数回の推論で速く安定しました。一方、CPUは推論の開始が早く(2～3秒)、推論時間も安定していました。物体検知中の推論時間と消費電力を比較すると、GPUとCPU、MAXNと5Wの違いが際立ちました(@<img>{image7})。推論時間は圧倒的にGPUの方が速いにもかかわらず、それほど消費電力の増加は見られませんでした。推論するのにGPUの方が電力効率が良いと言えます。

//image[image6][MAXNと5Wの消費電力]
//image[image7][MAXNと5Wの消費電力@物体検出中]