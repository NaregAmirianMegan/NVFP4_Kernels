import math

def compute_tmem_usage(m_tile_size):
	return max(32, m_tile_size*2) + 32

def compute_tiles(lst, N, m_tile_size, n_tile_size):
	total_tiles = 0
	n_tiles = N / n_tile_size
	for e in lst:
		total_tiles += math.ceil(e / m_tile_size)

	total_tiles *= n_tiles

	return total_tiles

def compute_waste(lst, N, n_tile_size, m_tile_size):
	n_tiles = N / n_tile_size
	waste = 0
	for e in lst:
		waste += e % m_tile_size

	return waste * n_tiles

def compute_pipe_stages_from_smem(num_ctas, m_tile_size, n_tile_size, k_tile_size, n_chunk_size, two_cta):
	total_smem = 230*1024
	total_smem -= (num_ctas * m_tile_size*n_chunk_size*2*2)
	pipe_stages = total_smem // (num_ctas * (m_tile_size*(k_tile_size/2)/(2 if two_cta else 1) + n_tile_size*(k_tile_size/2) + 2*(128*(k_tile_size/16))))
	return pipe_stages

if __name__ == "__main__":
	lst = [80, 176, 128, 72, 64, 248, 96, 160]
	N = 4096
	# lst = [40, 76, 168, 72, 164, 148, 196, 160]
	# N = 7168
	# lst = [192, 320]
	# N = 3072
	# lst = [128, 384]
	# N = 4096

	n_tile_size = 128
	m_tile_size = 128
	k_tile_size = 256
	n_chunk_size = 32
	two_cta = False
	num_ctas_per_sm = 1

	tmem_usage_per_cta = compute_tmem_usage(m_tile_size)
	if (512 // tmem_usage_per_cta) < num_ctas_per_sm:
		print("Too many ctas per sm")
		exit(0)

	tmem_util = ((num_ctas_per_sm * m_tile_size*2) / 512) * 100

	pipe_stages = compute_pipe_stages_from_smem(num_ctas_per_sm, m_tile_size, n_tile_size, k_tile_size, n_chunk_size, two_cta)

	total_tiles = compute_tiles(lst, N, m_tile_size, n_tile_size)

	waves = total_tiles / (num_ctas_per_sm * 148)

	wasted_tmem_compute = compute_waste(lst, N, n_tile_size, m_tile_size)

	print("num_ctas_per_sm:", num_ctas_per_sm)
	print("pipe_stages:", pipe_stages)
	print("total_tiles:", total_tiles)
	print("tmem_util:", tmem_util)
	print("waves:", waves)
	print("wasted_tmem_compute:", wasted_tmem_compute)





	"""
	First step is to implement transpose version that runs as fast as non-transpose version, then we can play around with sizes
	and implement 2CTA
	"""