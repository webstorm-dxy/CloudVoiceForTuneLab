# OpenUTAU 声库目录

把 **OpenUTAU 格式的 DiffSinger 声库**（每个声库一个文件夹）放在本目录下，服务端启动时会递归扫描
（默认深度 3，自动跳过 dsdur / dspitch / dsvariance / dsvocoder 等子模型目录）。

    OpenUTAU/
    ├── <声库名>/
    │   ├── character.txt          # 必需：name=...（OpenUTAU 声库标识）
    │   ├── character.yaml         # 可选：singer_type: diffsinger / subbanks / default_phonemizer
    │   ├── dsconfig.yaml          # 必需：acoustic / vocoder / phonemes / speakers / hop_size ...
    │   ├── dsdict.yaml            # 必需（或 dsdict-zh.yaml 等）：歌词->音素字典 + 音素类型表
    │   ├── phonemes.txt|json      # 必需：声学模型音素表（token id）
    │   ├── *.onnx                 # 声学模型
    │   ├── *.emb                  # 多说话人/音色的 speaker embedding（可选）
    │   ├── dsdur/                 # 可选：时长预测模型（有则音素时长更自然）
    │   │   ├── dsconfig.yaml
    │   │   ├── *.linguistic.onnx
    │   │   └── *.dur.onnx
    │   ├── dsvariance/            # 可选：方差模型（energy/breathiness/voicing/tension）
    │   └── dspitch/               # 可选：自动音高预测（当前版本暂不使用，只保留兼容）
    └── Vocoders/                  # 可选：内建声码器（nsf_hifigan 系列）
        └── nsf_hifigan/
            ├── vocoder.yaml
            └── *.onnx

## 声码器（nsf_hifigan）从哪来

OpenUTAU 的声库经常只写 vocoder: nsf_hifigan（内建名），模型本体在 OpenUTAU 的
Dependencies 目录里。服务端会自动在这些位置查找（也可用 --vocoder-dir 追加）：

- 本目录下的 Vocoders/（推荐：拷进来或建符号链接）；
- ~/Library/OpenUtau/Dependencies/ 与 ~/Documents/Library/OpenUtau/Dependencies/（macOS）；
- TuneLab 的 ChoristaDiffsinger 扩展 Vocoders/；
- 声库自带的 dsvocoder/（优先级最高，无需内建名）。

本机验证时可以直接建符号链接（不必拷 50MB+ 的模型）：

    cd demo/DiffSingerCloud/OpenUTAU/Vocoders
    ln -s "$HOME/Library/OpenUtau/Dependencies/nsf_hifigan_44.1k_hop512_128bin_2024.02" .
    ln -s "$HOME/Library/Application Support/TuneLab/Extensions/ChoristaDiffsinger/Vocoders/nsf_hifigan" .

## 本机已有的真实声库

macOS 上 OpenUTAU 的声库通常在 ~/Library/OpenUtau/Singers/。可以符号链接进来一起验证：

    cd demo/DiffSingerCloud/OpenUTAU
    ln -s "$HOME/Library/OpenUtau/Singers/JiHeHong-diffsinger-openutau-V2" .
    ln -s "$HOME/Library/OpenUtau/Singers/ChiYu" .

> 本目录下除 README.md 与 .gitignore 外的内容都被忽略，不会进版本库。
