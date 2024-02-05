#!/bin/bash

echo "--------------------------"
echo "install packages for build"
echo "--------------------------"
apt-get upgrade -y
apt-get update -y
apt-get install git -y
apt-get install curl -y
apt-get install gcc make -y
apt-get install -y net-tools tree vim telnet netcat
apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev wget llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev

echo "------------------------------------"
echo "install screen and env bashrc update"
echo "------------------------------------"
apt-get install screen -y
if !(ls -al ~/ | grep ".screenrc"); then
	echo 'ck 5000' >> ~/.screenrc
	echo 'vbell off' >> ~/.screenrc
	echo 'msgwait 3600' >> ~/.screenrc
	echo 'pow_detach_msg "Screen session of \$LOGNAME \$:cr:\$:nl:ended."' >> ~/.screenrc
	echo 'defhstatus "<^En-^Et> ^EW" # [^EM/^Ed(^ED) ^Ec]"' >> ~/.screenrc
	echo 'termcapinfo xterm Z0=\E[?3h:Z1=\E[?3l:is=\E[r\E[m\E[2J\E[H\E[?7h\E[?1;4;6l' >> ~/.screenrc
	echo 'termcapinfo xterm* OL=10000' >> ~/.screenrc
	echo 'termcapinfo xterm "VR=\E[?5h:VN=\E[?5l"' >> ~/.screenrc
	echo 'termcapinfo xterm "k1=\E[11~:k2=\E[12~:k3=\E[13~:k4=\E[14~"' >> ~/.screenrc
	echo 'termcapinfo xterm "kh=\E[1~:kI=\E[2~:kD=\E[3~:kH=\E[4~:kP=\E[H:kN=\E[6~"' >> ~/.screenrc
	echo 'termcapinfo xterm "vi=\E[?25l:ve=\E[34h\E[?25h:vs=\E[34l"' >> ~/.screenrc
	echo 'termcapinfo xterm "XC=K%,%\E(B,[\304,\\\\\326,]\334,{\344,|\366,}\374,~\337"' >> ~/.screenrc
	echo 'termcapinfo xterm ut' >> ~/.screenrc
	echo 'termcapinfo wy75-42 xo:hs@' >> ~/.screenrc
	echo 'termcapinfo wy* CS=\E[?1h:CE=\E[?1l:vi=\E[?25l:ve=\E[?25h:VR=\E[?5h:VN=\E[?5l:cb=\E[1K:CD=\E[1J' >> ~/.screenrc
	echo 'termcapinfo hp700 "Z0=\E[?3h:Z1=\E[?3l:hs:ts=\E[62"p\E[0$~\E[2$~\E[1$}:fs=\E[0}\E[61"p:ds=\E[62"p\E[1$~\E[61"p:ic@"' >> ~/.screenrc
	echo 'termcap vt100* ms:AL=\E[%dL:DL=\E[%dM:UP=\E[%dA:DO=\E[%dB:LE=\E[%dD:RI=\E[%dC' >> ~/.screenrc
	echo 'terminfo vt100* ms:AL=\E[%p1%dL:DL=\E[%p1%dM:UP=\E[%p1%dA:DO=\E[%p1%dB:LE=\E[%p1%dD:RI=\E[%p1%dC' >> ~/.screenrc
	echo 'defscrollback 10000' >> ~/.screenrc
	echo 'termcapinfo xterm* ti@:te@' >> ~/.screenrc
	echo 'startup_message off' >> ~/.screenrc
	echo 'hardstatus on' >> ~/.screenrc
	echo 'hardstatus alwayslastline' >> ~/.screenrc
	echo 'hardstatus string "%{ wk}%-w%{= kG}%n %t%{ kw}%+w %=%{ kB} %Y-%m-%d, %C %A, ${USER}@%H"' >> ~/.screenrc
	echo 'bindkey -k k1 select 0' >> ~/.screenrc
	echo 'bindkey -k k2 select 1' >> ~/.screenrc
	echo 'bindkey -k k3 select 2' >> ~/.screenrc
	echo 'defutf8 on' >> ~/.screenrc
fi

echo "------------------------------------------"
echo "install ko-language pack and bashrc update"
echo "------------------------------------------"
apt-get install language-pack-ko -y
if !(grep -qc "LANG" ~/.bashrc); then
	echo 'export LANG="ko_KR.UTF-8"' >> ~/.bashrc
fi

echo "-------------------------------"
echo "install pyenv and bashrc update"
echo "-------------------------------"
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
if !(grep -qc "PYENV_ROOT" ~/.bashrc); then
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
fi

sleep 5
. ~/.bashrc

echo "--------------------"
echo "install python3.11.4"
echo "--------------------"
pyenv install 3.11.4
pyenv global 3.11.4

echo "--------------------------------"
echo "install poetry and bashrc update"
echo "--------------------------------"
curl -sSL https://install.python-poetry.org | python3 -
if !(grep -qc "$HOME/.local/bin" ~/.bashrc); then
	echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo "Executing ~/.bashrc" >> ~/.bashrc
. ~/.bashrc