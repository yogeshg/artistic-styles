DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $DIR/profile.sh
nohup jupyter notebook >> logs/jupyter.log & echo $! >> $DIR/notebook.pid

