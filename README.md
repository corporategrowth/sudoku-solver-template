<<<<<<< HEAD
数独ソルバーの作成
===

## 準備

### GitHubのアカウントを作成

<https://github.com/>にアクセスしてアカウントを作成する。

### Gitクライアントのインストール (Windows向け)

Windowsの場合にはGitが最初からインストールされていないので、

- [Git for Windows](https://gitforwindows.org/)

を各自のコンピュータにインストールする。インストール後、コマンドプロンプトかPowerShellを実行して、

- `git`
- `ssh-keygen`

の2つのコマンドが認識されれば成功。

### SSHキーの登録

現在、GitHubはSSHの認証鍵を使わないとプライベートレポジトリをダウンロードできないので、SSHキーをGitHubアカウントに登録する。

Windows/Macともに、以下のコマンドで4096ビット長のRSA鍵を作成する。

```shell
ssh-keygen -t rsa -b 4096
```

途中、パスワードの入力などを求められるが、特に不要なら入力する必要はない。

コマンドが正しく実行されると、ホームディレクトリの`.ssh`ディレクトリ内に`id_rsa`と`id_rsa.pub`の二つのファイルが生成される。この二つのうち、`id_rsa`の方は秘密鍵、`id_rsa.pub`の方は公開鍵のファイルである。サーバーに登録して良いのは公開鍵の方。

公開鍵のファイル`id_rsa.pub`を何らかのエディタで開いて、その内容をコピーする。GitHubに移動し、右上のユーザアイコンをクリックし「Settings」を選ぶ。その後、「SSH and GPG keys」を左のメニューから選び、「SSH Keys」の右にある「New SSH key」ボタンを押して、現れるテキストボックスに先ほど`id_rsa.pub`からコピーした内容を貼り付けて、「Add SSH key」を押す。

### 課題用レポジトリの作成

講義中に指示する課題作成用URLにアクセスし、手順に従って、課題用のレポジトリである`sudoku-solver-username` (usernameの部分は各自のユーザ名)を作成する。

### レポジトリのクローン

再び、ローカルの環境に戻り、Gitレポジトリをクローンする。正しく、公開鍵が登録されていれば、以下のコマンドでレポジトリがクローンされる。

```shell
git clone git@github.com:tatsy-classes/sudoku-solver-username.git
```

### モジュールのインストール

Anacondaを使っている場合は適当な課題用の仮想環境を作成し、その環境下でPipを用いて必要なモジュールをインストールする

```shell
# 仮想環境の作成
conda create -n sudoku python
# 仮想環境の切り替え
conda activate sudoku
# モジュールのインストール
pip install -r requirements.txt
```


## 課題の作成方法

### ソルバー関数の編集

課題用レポジトリ (本レポジトリ)に含まれる `sudoku.py`を編集して、正しい数独の解が得られるプログラムに修正する。

### テスト方法

`data`ディレクトリの中に1枚ずつサンプルの画像が入っているのでそれを利用して良い。また、講義の参加者には`data/samples.zip`の展開用パスワードを指示するので、このZIPファイルに含まれる各レベル5枚のサンプル画像も合わせて使用して良い。準備ができたら、`pytest`を使ってテストを実行する。

```shell
# ログを全て表示する場合
pytest 
# 結果だけを表示する場合
pytest --tb=no -s
```
=======
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/uNAqkal6)
>>>>>>> origin/master
