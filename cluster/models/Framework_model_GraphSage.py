import torch.nn.functional as F
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys, os
import random
from collections import defaultdict
from sklearn.utils import shuffle
import math




class GraphSageModel:
    def __init__(self,  args):
        self.model_name=args.model_name
        self.args = args 

    def prepare(self):
        '''
        prepare dataloader, model, optimizer for training
        '''
        self.device=self.args.device
        dataSet='cora'
        agg_func="MEAN"
        self.epochs=self.args.total_epochs
        self.b_sz=self.args.batch_size
        seed=824
        self.learn_method='sup'
        self.unsup_loss='normal'
        self.ds=dataSet
        gcn=False
		
        random.seed(seed),
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
	
        self.dataCenter = DataCenter()
        self.dataCenter.load_dataSet(self.args.dataset_dir,dataSet)
        # self.dataCenter = DataLoader(self.dataCenter, sampler=DistributedSampler(self.dataCenter), batch_size=self.args.batch_size)
        
        features = torch.FloatTensor(getattr(self.dataCenter, dataSet+'_feats')).to(self.device)

        graphSage = GraphSage(self.args.layer_num, features.size(1), self.args.layer_feature , features, getattr(self.dataCenter, dataSet+'_adj_lists'), self.device, gcn=gcn, agg_func=agg_func)
        graphSage.to(self.device)
        self.graphSage=DDP(graphSage, device_ids=[self.device],output_device=self.device)
		

        num_labels = len(set(getattr(self.dataCenter, dataSet+'_labels')))
        classification = Classification(self.args.layer_feature, num_labels)
        classification.to(self.device)
        self.classification=DDP(classification, device_ids=[self.device],output_device=self.device)

        self.unsupervised_loss = UnsupervisedLoss(getattr(self.dataCenter, dataSet+'_adj_lists'), getattr(self.dataCenter, dataSet+'_train'), self.device)
        
        print("end ddp model...")
        
        self.graphSage.train()
        self.cur_epoch = 0
        train_nodes=getattr(self.dataCenter, self.ds+'_train')
        self.total_batch_num=math.ceil(len(train_nodes) / self.b_sz)
    
    
    def prepare_sub(self): 
        
        
        self.train_nodes = getattr(self.dataCenter, self.ds+'_train')
        self.labels = getattr(self.dataCenter, self.ds+'_labels')

        if self.unsup_loss == 'margin':
            self.num_neg = 6
        elif self.unsup_loss == 'normal':
            self.num_neg = 100
        else:
            print("unsup_loss can be only 'margin' or 'normal'.")
            sys.exit(1)

        self.train_nodes = shuffle(self.train_nodes)

        self.models = [self.graphSage, self.classification]
        self.params = []
        for model in self.models:
            for param in model.parameters():
                if param.requires_grad:
                    self.params.append(param)

        self.optimizer = torch.optim.SGD(self.params, lr=0.7)
        self.optimizer.zero_grad()
        for model in self.models:
            model.zero_grad()

        self.visited_nodes = set()
        self.batch_idx = 0

    def is_epoch_end(self):
        if self.batch_idx==self.total_batch_num-1:
            return True
        else:
            return False
        
    def is_end(self):
        if self.cur_epoch==self.args.total_epoch_num-1 and self.batch_idx==self.total_batch_num-1:
            return True
        else:
            return False
        
        
    def get_data(self):
        '''
        get data
        '''
        if self.batch_idx<self.total_batch_num:
            nodes_batch = self.train_nodes[self.batch_idx*self.b_sz:(self.batch_idx+1)*self.b_sz]
            nodes_batch = np.asarray(list(self.unsupervised_loss.extend_nodes(nodes_batch, num_neg=self.num_neg)))
        else:
            self.cur_epoch += 1
            self.prepare_sub()
            
        self.batch_idx +=1
        
        return nodes_batch
    
    def forward_backward(self, nodes_batch):
        '''
        forward, calculate loss and backward
        '''
        self.visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = self.labels[nodes_batch]
        embs_batch = self.graphSage(nodes_batch)

        if self.learn_method == 'sup':
            # superivsed learning
            logists = self.classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            loss = loss_sup
        elif self.learn_method == 'plus_unsup':
            # superivsed learning
            logists = self.classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if self.unsup_loss == 'margin':
                loss_net = self.unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif self.unsup_loss == 'normal':
                loss_net = self.unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            if self.unsup_loss == 'margin':
                loss_net = self.unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif self.unsup_loss == 'normal':
                loss_net = self.unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net

        
        loss.backward()
        for model in self.models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        

    def comm(self):
        '''
        sync for communication
        '''
        self.optimizer.step()

        self.optimizer.zero_grad()
        for model in self.models:
            model.zero_grad()
    
    
    
    def sample(self):
        self.train_nodes = getattr(self.dataCenter, self.ds+'_train')
        self.labels = getattr(self.dataCenter, self.ds+'_labels')

        if self.unsup_loss == 'margin':
            self.num_neg = 6
        elif self.unsup_loss == 'normal':
            self.num_neg = 100
        else:
            print("unsup_loss can be only 'margin' or 'normal'.")
            sys.exit(1)

        self.train_nodes = shuffle(self.train_nodes)

        self.models = [self.graphSage, self.classification]
        self.params = []
        for model in self.models:
            for param in model.parameters():
                if param.requires_grad:
                    self.params.append(param)

        self.optimizer = torch.optim.SGD(self.params, lr=0.7)
        self.optimizer.zero_grad()
        for model in self.models:
            model.zero_grad()

        self.visited_nodes = set()

        
        
    def train(self):

        for index in range(self.total_batch_num):
            if index%500 == 0:
                    print(f"job_idx: {self.args.job_idx} batch_idx: {index}/{self.total_batch_num}...")
                
                
            nodes_batch = self.train_nodes[index*self.b_sz:(index+1)*self.b_sz]

            # extend nodes batch for unspervised learning
            # no conflicts with supervised learning
            nodes_batch = np.asarray(list(self.unsupervised_loss.extend_nodes(nodes_batch, num_neg=self.num_neg)))
            self.visited_nodes |= set(nodes_batch)

            # get ground-truth for the nodes batch
            labels_batch = self.labels[nodes_batch]
            embs_batch = self.graphSage(nodes_batch)

            if self.learn_method == 'sup':
                # superivsed learning
                logists = self.classification(embs_batch)
                loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
                loss_sup /= len(nodes_batch)
                loss = loss_sup
            elif self.learn_method == 'plus_unsup':
                # superivsed learning
                logists = self.classification(embs_batch)
                loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
                loss_sup /= len(nodes_batch)
                # unsuperivsed learning
                if self.unsup_loss == 'margin':
                    loss_net = self.unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
                elif self.unsup_loss == 'normal':
                    loss_net = self.unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
                loss = loss_sup + loss_net
            else:
                if self.unsup_loss == 'margin':
                    loss_net = self.unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
                elif self.unsup_loss == 'normal':
                    loss_net = self.unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
                loss = loss_net

            
            loss.backward()
            for model in self.models:
                nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()

            self.optimizer.zero_grad()
            for model in self.models:
                model.zero_grad()
            
            





        
'''***********************************************datacenter******************************************************************'''

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self):
		super(DataCenter, self).__init__()

		
	def load_dataSet(self, data_dir,dataSet='cora'):
		if dataSet == 'cora':
			cora_content_file = "cora.content"
			cora_cite_file = "cora.cites"

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			label_map = {} # map label to Label_ID
			# cur_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


			with open(os.path.join(data_dir,"cora",cora_content_file) ) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					feat_data.append([float(x) for x in info[1:-1]])
					node_map[info[0]] = i
					if not info[-1] in label_map:
						label_map[info[-1]] = len(label_map)
					labels.append(label_map[info[-1]])
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(os.path.join(data_dir,"cora",cora_cite_file)) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 2
					paper1 = node_map[info[0]]
					paper2 = node_map[info[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)

			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test',  DataLoader(test_indexs, sampler=DistributedSampler(test_indexs)).dataset)
			setattr(self, dataSet+'_val', DataLoader(val_indexs, sampler=DistributedSampler(val_indexs)).dataset )
			setattr(self, dataSet+'_train', DataLoader(train_indexs, sampler=DistributedSampler(train_indexs)).dataset )

			setattr(self, dataSet+'_feats', DataLoader(feat_data, sampler=DistributedSampler(feat_data)).dataset )
			setattr(self, dataSet+'_labels', DataLoader(labels, sampler=DistributedSampler(labels)).dataset )
			setattr(self, dataSet+'_adj_lists', DataLoader(adj_lists, sampler=DistributedSampler(adj_lists)).dataset )



		elif dataSet == 'pubmed':
			pubmed_content_file = "Pubmed-Diabetes.NODE.paper.tab"
			pubmed_cite_file = "Pubmed-Diabetes.DIRECTED.cites.tab"

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			with open(os.path.join(data_dir,"pubmed-data",pubmed_content_file)) as fp:
				fp.readline()
				feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
				for i, line in enumerate(fp):
					info = line.split("\t")
					node_map[info[0]] = i
					labels.append(int(info[1].split("=")[1])-1)
					tmp_list = np.zeros(len(feat_map)-2)
					for word_info in info[2:-1]:
						word_info = word_info.split("=")
						tmp_list[feat_map[word_info[0]]] = float(word_info[1])
					feat_data.append(tmp_list)
			
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(os.path.join(data_dir,"pubmed-data",pubmed_cite_file)) as fp:
				fp.readline()
				fp.readline()
				for line in fp:
					info = line.strip().split("\t")
					paper1 = node_map[info[1].split(":")[1]]
					paper2 = node_map[info[-1].split(":")[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)
			
			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)


	def _split_data(self, num_nodes, test_split = 3, val_split = 6):
		rand_indices = np.random.permutation(num_nodes)

		test_size = num_nodes // test_split
		val_size = num_nodes // val_split
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[:test_size]
		val_indexs = rand_indices[test_size:(test_size+val_size)]
		train_indexs = rand_indices[(test_size+val_size):]
		
		return test_indexs, val_indexs, train_indexs


'''**********************************************model***********************************************************************'''
class GraphSage(nn.Module):
	"""docstring for GraphSage"""
	def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
		super(GraphSage, self).__init__()

		self.input_size = input_size
		self.out_size = out_size
		self.num_layers = num_layers
		self.gcn = gcn
		self.device = device
		self.agg_func = agg_func

		self.raw_features = raw_features
		self.adj_lists = adj_lists

		for index in range(1, num_layers+1):
			layer_size = out_size if index != 1 else input_size
			setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

	def forward(self, nodes_batch):
		"""
		Generates embeddings for a batch of nodes.
		nodes_batch	-- batch of nodes to learn the embeddings
		"""
		lower_layer_nodes = list(nodes_batch)
		nodes_batch_layers = [(lower_layer_nodes,)]
		# self.dc.logger.info('get_unique_neighs.')
		for i in range(self.num_layers):
			lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)
			nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

		assert len(nodes_batch_layers) == self.num_layers + 1

		pre_hidden_embs = self.raw_features
		for index in range(1, self.num_layers+1):
			nb = nodes_batch_layers[index][0]
			pre_neighs = nodes_batch_layers[index-1]
			# self.dc.logger.info('aggregate_feats.')
			aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
			sage_layer = getattr(self, 'sage_layer'+str(index))
			if index > 1:
				nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
			# self.dc.logger.info('sage_layer.')
			cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
										aggregate_feats=aggregate_feats)
			pre_hidden_embs = cur_hidden_embs

		return pre_hidden_embs

	def _nodes_map(self, nodes, hidden_embs, neighs):
		layer_nodes, samp_neighs, layer_nodes_dict = neighs
		assert len(samp_neighs) == len(nodes)
		index = [layer_nodes_dict[x] for x in nodes]
		return index

	def _get_unique_neighs_list(self, nodes, num_sample=10):
		_set = set
		to_neighs = [self.adj_lists[int(node)] for node in nodes]
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs
		samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
		_unique_nodes_list = list(set.union(*samp_neighs))
		i = list(range(len(_unique_nodes_list)))
		unique_nodes = dict(list(zip(_unique_nodes_list, i)))
		return samp_neighs, unique_nodes, _unique_nodes_list

	def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
		unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

		assert len(nodes) == len(samp_neighs)
		indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
		assert (False not in indicator)
		if not self.gcn:
			samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]
		# self.dc.logger.info('2')
		if len(pre_hidden_embs) == len(unique_nodes):
			embed_matrix = pre_hidden_embs
		else:
			embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
		# self.dc.logger.info('3')
		mask = torch.zeros(len(samp_neighs), len(unique_nodes))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		# self.dc.logger.info('4')

		if self.agg_func == 'MEAN':
			num_neigh = mask.sum(1, keepdim=True)
			mask = mask.div(num_neigh).to(embed_matrix.device)
			aggregate_feats = mask.mm(embed_matrix)

		elif self.agg_func == 'MAX':
			# print(mask)
			indexs = [x.nonzero() for x in mask==1]
			aggregate_feats = []
			# self.dc.logger.info('5')
			for feat in [embed_matrix[x.squeeze()] for x in indexs]:
				if len(feat.size()) == 1:
					aggregate_feats.append(feat.view(1, -1))
				else:
					aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
			aggregate_feats = torch.cat(aggregate_feats, 0)

		# self.dc.logger.info('6')
		
		return aggregate_feats






class UnsupervisedLoss(object):
	"""docstring for UnsupervisedLoss"""
	def __init__(self, adj_lists, train_nodes, device):
		super(UnsupervisedLoss, self).__init__()
		self.Q = 10
		self.N_WALKS = 6
		self.WALK_LEN = 1
		self.N_WALK_LEN = 5
		self.MARGIN = 3
		self.adj_lists = adj_lists
		self.train_nodes = train_nodes
		self.device = device

		self.target_nodes = None
		self.positive_pairs = []
		self.negtive_pairs = []
		self.node_positive_pairs = {}
		self.node_negtive_pairs = {}
		self.unique_nodes_batch = []

	def get_loss_sage(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			# Q * Exception(negative score)
			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
			#print(neg_score)

			# multiple positive score
			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score = torch.log(torch.sigmoid(pos_score))
			#print(pos_score)

			nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))
				
		loss = torch.mean(torch.cat(nodes_score, 0))
		
		return loss

	def get_loss_margin(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

			nodes_score.append(torch.max(torch.tensor(0.0).to(self.device), neg_score-pos_score+self.MARGIN).view(1,-1))
			# nodes_score.append((-pos_score - neg_score).view(1,-1))

		loss = torch.mean(torch.cat(nodes_score, 0),0)

		# loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))
		
		return loss


	def extend_nodes(self, nodes, num_neg=6):
		self.positive_pairs = []
		self.node_positive_pairs = {}
		self.negtive_pairs = []
		self.node_negtive_pairs = {}

		self.target_nodes = nodes
		self.get_positive_nodes(nodes)
		# print(self.positive_pairs)
		self.get_negtive_nodes(nodes, num_neg)
		# print(self.negtive_pairs)
		self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
		assert set(self.target_nodes) < set(self.unique_nodes_batch)
		return self.unique_nodes_batch

	def get_positive_nodes(self, nodes):
		return self._run_random_walks(nodes)

	def get_negtive_nodes(self, nodes, num_neg):
		for node in nodes:
			neighbors = set([node])
			frontier = set([node])
			for i in range(self.N_WALK_LEN):
				current = set()
				for outer in frontier:
					current |= self.adj_lists[int(outer)]
				frontier = current - neighbors
				neighbors |= current
			far_nodes = set(self.train_nodes) - neighbors
			neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
			self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
			self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
		return self.negtive_pairs

	def _run_random_walks(self, nodes):
		for node in nodes:
			if len(self.adj_lists[int(node)]) == 0:
				continue
			cur_pairs = []
			for i in range(self.N_WALKS):
				curr_node = node
				for j in range(self.WALK_LEN):
					neighs = self.adj_lists[int(curr_node)]
					next_node = random.choice(list(neighs))
					# self co-occurrences are useless
					if next_node != node and next_node in self.train_nodes:
						self.positive_pairs.append((node,next_node))
						cur_pairs.append((node,next_node))
					curr_node = next_node

			self.node_positive_pairs[node] = cur_pairs
		return self.positive_pairs
		

class SageLayer(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, input_size, out_size, gcn=False): 
		super(SageLayer, self).__init__()

		self.input_size = input_size
		self.out_size = out_size


		self.gcn = gcn
		self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

		self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, self_feats, aggregate_feats, neighs=None):
		"""
		Generates embeddings for a batch of nodes.

		nodes	 -- list of nodes
		"""
		if not self.gcn:
			combined = torch.cat([self_feats, aggregate_feats], dim=1)
		else:
			combined = aggregate_feats
		combined = F.relu(self.weight.mm(combined.t())).t()
		return combined



class Classification(nn.Module):

	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)	  
								#nn.ReLU()
							)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists