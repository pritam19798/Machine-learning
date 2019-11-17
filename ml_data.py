import numpy as np
observaton=1000

xs=np.random.uniform(low=-10,high=10,size=(observaton,1))
zs=np.random.uniform(low=-10,high=10,size=(observaton,1))

genatated_input=np.column_stack((xs,zs))
noise=np.random.uniform(-1,-1,(observaton,1))

generted_target=2*xs-3*zs+5+noise

np.savez('my_data',inputs=genatated_input,target=generted_target)
