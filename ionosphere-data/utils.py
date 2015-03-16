from random import shuffle

def to_pure_numbers():
	f = open('ionosphere.data', 'r')
	set_two_raw = [ line.replace('g', '1').replace('b', '0') for line in f.readlines() ]
	f.close()

	shuffle(set_two_raw) # randomly arranges the instances so they can be split

	f = open('ionosphere.txt', 'w')
	for line in set_two_raw:
		f.write(line)
	f.close()

if __name__ == '__main__':
	to_pure_numbers()