= NVIDIA Jetson Nanoの消費電力

//flushright{
Seeed 松岡、中井
//}

AI界隈、活況ですね。MidjourneyやStable Diffusionなどの画像生成で盛り上がったかと思うと、今度はAIチャットサービスChatGPTの話でもちきりです。ワクワクしますね。ハードウェアを設計製造している弊社SeeedでもAI関連のハードウェアをいくつか提供しています。最近はNVIDIA Jetsonシリーズのキャリアボードやボックス型コンピュータのラインナップをもりもり増やしています。

NVIDIA Jetsonと聞くと、Jetson Nano Developer Kit（@<img>{1-JetsonNanoDeveloperKit}）を思い浮かべる方が多いのではないでしょうか。2019年に販売開始したNVIDIA製GPUを備えた組み込みシステム向けシングルボードコンピュータで、NVIDIAプラットフォームで作られたAIソフトウェアを動かすことができます。日本国内での入手性も良く、$100を切る低価格だったことから、多くの方々が手に取って動かし楽しんだと思います。

//image[1-JetsonNanoDeveloperKit][Jetson Nano Developer Kit]

このNVIDIA Jetson、特徴の１つに消費電力の少なさがあります。一般的なGPUは並列実行で膨大な量の計算を速く処理することを目指していますが、大量な電力を必要とするので組み込みシステムに少し不向きです。自走式ロボットなど、電源がバッテリーで供給されていて、使うことが出来る電力が制限されているときがあるからです。このような用途にNVIDIA Jetsonは最適です。「Jetson is compatible with the same AI software and cloud-native workflows used across other NVIDIA platforms and delivers the @<em>{power-efficient} performance customers need to build software-defined autonomous machines」（https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/）とあるように電力効率を重視して作られていて、5~60Wで動くモジュールがラインナップされています@<img>{2-JetsonModules}。そこで今回は（SeeedのreComputer J1020を使って）NVIDIA Jetsonモジュールの中で最も消費電力の少ないJetson Nanoモジュールの消費電力を確認してみようと思います。

//image[2-JetsonModules][NVIDIA Jetsonモジュールと消費電力]

== Jetson Nanoモジュールの消費電力を見てみよう

reComputer J1020（https://www.seeedstudio.com/Jetson-10-1-H0-p-5335.html）は製品版Jetson Nanoモジュールをキャリアボードに載せてアルミケースに入れた製品です(@<img>{3-J1020})。あえて“製品版”と書いているのはJetson Nano Developer KitやJetson Nano 2GB Developer Kitとは違うからです（Developer Kitに載っているJetson NanoモジュールはeMMCが無かったり、メモリが少なかったりします）。reComputer J1020を動かすには追加でディスプレイとキーボード、マウスが必要です。また、販売ルートによってACアダプタが付属していたり無かったりします。付属していてもPSE認証が無いACアダプタのときは使ってはいけません。9~19VのACアダプタ（2.1mm/5.5mmプラグ、センタープラス）を追加で用意します。

//image[3-J1020][reComputer J1020]


実験はreComputer J1020にUSBカメラを付けてYOLOv7を使って物体検出（Object Detection）しているときの消費電力を測ることにします(@<img>{4-TestBench2})。機材は@<img>{5-Parts}のとおりです。reComputer J1020にACアダプタ、USBカメラ、ディスプレイ、キーボード、マウスをつなぎます。ストレージとして256GB NVMe SSDを追加します。Jetson Nanoモジュールに内蔵している16GB eMMCだけではYOLOv7を動かすには容量が足りないからです。Jetson Nanoモジュールの放熱がやや不安なので、ヒートシンクをSeeedのアクティブヒートシンクに交換しておきます。Jetson Nanoモジュールが高温になるとヒートシンクに付いているファンが回って強制冷却してくれるので、少し安心です。

//image[4-TestBench2][実験の様子]

//image[5-Parts][実験に使う機材]

次はソフトウェアです。ブログ（https://lab.seeed.co.jp/entry/2022/06/06/120000）を参考に、eMMCへJetson LinuxとJetson stats、JetPackをインストールします。（執筆時点のJetson Linux最新バージョンは32.7.3。）次に、NVMe SSDをフォーマットして各種パッケージをインストール、セットアップします。

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
$ wget https://nvidia.box.com/shared/static/2sv2fv1wseihaw8ym0d4srz41dzljwxh.whl -O 
onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl
$ python -m pip install onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl
$ rm onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl
$ python -m pip install opencv-python

# モデルファイルをダウンロード
$ cd models
$ wget https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/main/307_YOLOv7/
download_single_batch.sh
$ chmod +x ./download_single_batch.sh
$./download_single_batch.sh
$ cd ..
//}

準備が整ったので、物体検出を実行しましょう。gishohaku8ディレクトリでPython仮想環境をアクティブにしてからobject_detection.pyを起動します。新しくウィンドウが開いてカメラの映像と物体検出のバウンディングボックスが表示されれば正常です。object_detection.pyの引数に”-provider cuda”を指定すると推論（Inference）にGPUを使い、”-provider cpu”を指定するとCPUを使います。

//emlist{
# 初回のみ
$ cd /mnt/ssd/gishohaku8
$ source venv/bin/activate
$ export DISPLAY=:0.0

$ python ./object_detection.py –provider cuda     # GPUで計算
$ python ./object_detection.py –provider cpu      # CPUで計算
//}

物体検出中の消費電力はJetson Nanoモジュールに内蔵しているINA3221というセンサで測ります。tegrastatsコマンドやjtopコマンドで表示される電力も、このセンサの値です。jpower.shを起動すると定期的にセンサを読み取ってタイムスタンプと消費電力を表示します。表示されるPOM_5V_IN、POM_5V_GPU、POM_5V_CPUは、ミリワット単位のモジュール全体の消費電力、GPUの消費電力、CPUの消費電力を示しています。

//emlist{
$ sudo /mnt/ssd/gishohaku8/jpower.sh
//}


パワーモード（後記）をMAXNと5WにしてGPUとCPUで物体検出しているときの消費電力を測った結果が@<img>{6-graph1}です。GPUはobject_detection.pyを起動してから推論が始まるまで少し時間がかかっていました。10～15秒ほど。また、初回の推論時間が遅く（4～6秒）、数回推論すると速く安定しました。どうやら、動き始めはGPU周波数が最大まで上がっていないようです。一方、CPUは推論の開始が早く（2～3秒）、推論時間は一定でした。物体検知中の推論時間と消費電力をグラフにしたのが@<img>{7-graph2}です。推論時間は圧倒的にGPUの方が速い（7~15倍）にもかかわらず、消費電力はたいして増加（1.4~1.5倍）しませんでした。電力効率はGPUの方が良いと言えるでしょう。


//image[6-graph1][MAXNと5Wの消費電力]
//image[7-graph2][MAXNと5Wの消費電力@物体検出中]

== パワーモードの切り替え

実はJetson NanoモジュールはCPUを無効化したり動作周波数を下げる仕組みを備えています。これらを調整することで、意図的に動作を遅くして用途に応じた消費電力に抑えることが可能です。具体的には、/etc/nvpmodel.confに書いておいた設定にnvpmodelコマンドで切り替えます。この設定名をパワーモードと言います。デフォルトでMAXNと5Wの2つのパワーモードが用意されていて、”--query”で現在のパワーモードを表示、”--mode”でパワーモードを切り替えることができます。

//emlist{
$ sudo nvpmodel --query
$ sudo nvpmodel --mode （パワーモードID）
//}

パワーモードの設定内容は/etc/nvpmodel.confに書かれています。デフォルトのMAXNと5Wを比較すると、5WではCPUコアを2つ停止して、CPUとGPUの上限周波数を6～7割に抑えていることが分かります。

//emlist{
$ cat /etc/nvpmodel.conf
...
< POWER_MODEL ID=0 NAME=MAXN >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 1
CPU_ONLINE CORE_3 1
CPU_A57 MIN_FREQ  0
CPU_A57 MAX_FREQ -1	# 1479000[Hz]
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
GPU MIN_FREQ  0
GPU MAX_FREQ -1	# 921600000[Hz]
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ 0

< POWER_MODEL ID=1 NAME=5W >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 0
CPU_ONLINE CORE_3 0
CPU_A57 MIN_FREQ  0
CPU_A57 MAX_FREQ 918000
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
GPU MIN_FREQ 0
GPU MAX_FREQ 640000000
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ 1600000000
...
//}

== もっと消費電力を抑えたい
パワーモード5Wの周波数をさらに下げることでもっと消費電力を抑えることができるのではないか？と思い、設定変更を試みることにしました。

周波数はどんな値にでも変更できるわけではなく、特定の値に限定されているようです。設定可能なGPU周波数は/sys/devices/gpu.0/devfreq/57000000.gpu/available_frequenciesで確認できました。

//emlist{
$ cat /sys/devices/gpu.0/devfreq/57000000.gpu/available_frequencies | tr ' ' '\n'
76800000
153600000
230400000
307200000
384000000
460800000
537600000
614400000
691200000
768000000
844800000
921600000	# MAXN
//}

パワーモード5Wで設定している640000000が見つかりません。謎い。パワーモードを5Wにしておき、GPU周波数をjetson_clocksコマンドで確認したところ614400000でした。

//emlist{
$ sudo jetson_clocks --show
SOC family:tegra210  Machine:NVIDIA Jetson Nano Developer Kit
Online CPUs: 0-1
cpu0: Online=1 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 
　IdleStates: WFI=0 c7=0
cpu1: Online=1 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 
　IdleStates: WFI=0 c7=0
cpu2: Online=0 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 
　IdleStates: WFI=0 c7=0
cpu3: Online=0 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 
　IdleStates: WFI=0 c7=0
GPU MinFreq=614400000 MaxFreq=614400000 CurrentFreq=614400000
EMC MinFreq=204000000 MaxFreq=1600000000 CurrentFreq=1600000000 FreqOverride=1
Fan: PWM=0
NV Power Mode: 5W
//}

614400000未満のGPU周波数で調べることにしましょう。パワーモード5WのGPU MAX_FREQを76800000~537600000に変えて、物体検出しているときの消費電力を測りました（@<img>{8-graph3}）。周波数を下げると推論が遅くなり、消費電力が下がりました。GPU周波数が300000000あたりから下になると消費電力の減少は鈍くなり、かつ、推論は大幅に遅くなりました。下の方は効率が急激に悪化するようです。消費電力約4000ミリワット、推論時間約200ミリ秒のGPU周波数384000000がもっとも電力効率が良さそうです。

//image[8-graph3][GPU周波数変更の消費電力 @物体検出中][scale=1.0]

== 単三乾電池で動くのか！？
ここまで消費電力が低いならば、もしかすると乾電池でそれなりの時間稼働できるかもしれません。実験してみましょう。

reComputer J1020の電源は9~19Vに対応しています。アルカリ乾電池の解放電圧は約1.6Vなので、直接に接続しても安全な本数は11本以下です。入手できる電池ボックスに10本入るものがあったので、今回の実験は@<em>{アルカリ乾電池 単三型を10本}とします。電池ボックスにDCジャックを結線してreComputer J1020につなげれるようにします。できるだけ電力を消費しないよう、周辺機器のディスプレイ、キーボード、マウスは外します。有線LANも接続しません。コンソールはPCからUSB-UART変換ケーブルを接続して操作します（@<img>{9-Battery2}）。パワーモードの設定は5WをベースにGPU周波数を384000000にします。

//image[9-Battery2][単三乾電池で実験している様子][scale=1.0]

物体検出の実行は、ディスプレイに画像を表示しないヘッドレスモードで起動します。

//emlist{
$ python ./object_detection.py --provider cuda --headless
- provider: cuda
- model: models/yolov7-tiny_384x640.onnx
- interval-ms: 0
- headless: True
Inference: 1682424716.6073656 90.894273245 9978.39
Inference: 1682424716.9537761 91.240675484 294.46
Inference: 1682424717.196005 91.48290137 188.15
Inference: 1682424717.4386811 91.725577984 190.16
...
//}

このまま動かし続けたところ、@<em>{2時間46分後}に乾電池がカラになって止まりました。数十分しか動かないと想像していたので驚きの結果です。単二型や単一型を使えば、ちょっとした実験に使えそうですね。

== Jetson Xavier NXモジュールの消費電力
Jetson Nanoモジュールより高性能なJetson Xavier NXを搭載したreComputer J2011（https://www.seeedstudio.com/Jetson-20-1-H1-p-5328.html）が手元にあったので、GPU周波数を変えて測りました（@<img>{10-graph4}）。Jetson Nanoモジュールに比べ、推論がかなり速いですが、消費電力が大きいです。周波数を下げても7000mWぐらいまでしか減らず、同レベルのJetson Nanoモジュールの推論時間よりも遅い結果でした。推論時間が遅くても良いときは、Jetson Nanoモジュールを使用したほうが省電力なのがわかりました。

//image[10-graph4][Jetson NanoモジュールとJetson Xavier NXモジュールの消費電力 @物体検出中]

== SeeedのNVIDIA Jetson製品
最後に宣伝です。Seeedからは、今回実験に使ったreComputer J1020、reComputer J2011以外のNVIDIA Jetson製品も製造販売しています。モジュールの種類は豊富（Jetson Nano/Xavier NX/Orin Nano/Orin NX）で、さまざまな用途に使いやすいようキャリアボード、ハンドサイズエッジAIボックス、産業用セットトップボックス、エッジAIサーバを提供しています（@<img>{11-SeeedJetsonDevices}）。また、これらの製品をお客様の目的に最適になるようカスタマイズするサービスも用意しています。詳しくは、2023年の製品発表イベントページ「MAKE SENSE FROM THE TRUE WILD」（https://2023.seeed.cc）の下の方にあるSeeed Studio Catalog 2023をご参照ください。

//image[11-SeeedJetsonDevices][SeeedのNVIDIA Jetson製品][scale=1.0]