export HYDRA_FULL_ERROR=1
export expname='xxx'
## not provide L
python main.py data=v241113 model=flow expname=${expname} 

## provide L
python main.py data=v241113 model=flow model.cost_lattice=0 expname=${expname} 

