DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PATH="/home/ubuntu/miniconda2/bin:$PATH"
source activate theano
nohup jupyter notebook >> logs/jupyter.log & echo $! >> $DIR/notebook.pid

