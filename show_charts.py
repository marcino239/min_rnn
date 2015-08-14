import re
import sys
import matplotlib.pyplot as plt

LINE_SUBS = [ 'iter', 'loss' ]

def process_log( file_name ):
	
	fin = open( file_name, 'rt' )
	iterations, loss = [], []
	
	for line in fin:
		if all( x in line for x in LINE_SUBS ):
			temp = re.split( ',| ', line )
			if len( temp ) == 5:
				iterations.append( float( temp[1] ) )
				loss.append( float( temp[4] ) )
	
	fin.close()
	return iterations, loss


relu_i, relu_l = process_log( sys.argv[1] )
tanh_i, tanh_l = process_log( sys.argv[2] )

fig, ax = plt.subplots( figsize=(10,6) )

ax.plot( relu_i, relu_l, 'b-', label = sys.argv[1] )
ax.plot( tanh_i, tanh_l, 'g-', label = sys.argv[2] )
legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')

ax.set_ylabel( 'loss' )
ax.set_xlabel( 'iterations' )

plt.show()
