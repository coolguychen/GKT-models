import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from layers import MLP, EraseAddGate, MLPEncoder, MLPDecoder, ScaledDotProductAttention

# Graph-based knowledge tracing
# 自己手写代码

class GKT():
    # concept_num:概念数；hidden_dim:隐藏层维度，embedding_dim:嵌入层维度；edge_type_num:边类型数量；graph_type:图结构类型
    # graph:图结构；graph_model:图模型 dropout：dropout概率，bias：偏置项；binary：二元分类；hascuda：是否使用gpu
    def __init__(self, concept_num, hidden_dim, embedding_dim, edgt_type_num, graph_type,):
        def __init__(self, concept_num, hidden_dim, embedding_dim, edge_type_num, graph_type, graph=None,
                     graph_model=None, dropout=0.5, bias=True, binary=False, has_cuda=False):
            super(GKT, self).__init__()
            self.concept_num = concept_num # 知识点数目
            self.hidden_dim = hidden_dim # 隐藏层维度
            self.embedding_dim = embedding_dim # 嵌入层维度
            self.edge_type_num = edge_type_num # 边数目 == 2

            # 将输入的边类型数量赋值给self.~
            # res_len表示每个知识点的反应次数，或者说，每个题的回答历史情况；具体而言，如果二进制，使用0，1分别表示正确和错误，则知识点可能有两次反应，如果采用多元分类，则一个知识点可能有多个反应
            self.res_len = 2 if binary else 12 # why 12?
            self.has_cuda = has_cuda # 是否使用GPU

            # python中的assert关键字是用来检查一个条件，如果为真，就不做任何事。如果它是假，则会抛出AssertError并且包含错误信息。
            # assert格式：assert expression[,arguments]；expression是检查的条件表达式，如果返回值为false，则会引发异常输出arguments参数的信息
            assert graph_type in ['Dense', 'Transition', 'DKT', 'PAM', 'MHA', 'VAE'] # 检查 graph_type是否在可选值中，如果条件为假，会引发assertionError异常
            self.graph_type = graph_type # 图类型赋值
            if graph_type in ['Dense', 'Transition', 'DKT']:
                # 检查边类型数量是否为2 （由于是dense、transition、dkt这几个图，则要求边类型数量为2）
                assert edge_type_num == 2
                # 检查图是否参数（not None）而不提供图模型参数 但是传入graph==None?
                assert graph is not None and graph_model is None
                # 将图参数graph转换为可学习的参数nn.parameter,图形状为[知识点数目, 知识点数目]
                self.graph = nn.Parameter(graph)  # [concept_num, concept_num]
                # 固定self.graph参数，使其在反向传播时不会被更新
                self.graph.requires_grad = False  # fix parameter
                self.graph_model = graph_model # 设置图模型
            else:  # ['PAM', 'MHA', 'VAE']
                assert graph is None # 检查是否未提供图类型
                self.graph = graph  # None
                if graph_type == 'PAM':
                    # 如果使用的是PAM，不需要图模型，生成随机参数矩阵graph
                    assert graph_model is None
                    self.graph = nn.Parameter(torch.rand(concept_num, concept_num))
                else: # 如果是MHA和VAE
                    # 检查要求提供图模型
                    assert graph_model is not None
                self.graph_model = graph_model # 图模型赋值

            # one-hot feature and question
            # 生成one-hot特征向量，大小是[反应次数*概念数，反应次数*概念数]的单位矩阵（对角线上全为1）
            one_hot_feat = torch.eye(self.res_len * self.concept_num) # torch.eye生成对角线全1的矩阵
            '''
            若res_len == 2,也就是反应次数为2，概念上concept_num == 4，one_hot_feat = [8,8]
                  1,2,3,4,1',2',3',4'
               1 [1,0,0,0,0,0,0,0;
               2  0,1,0,0,0,0,0,0;
               3  0,0,1,0,0,0,0,0;
               4  0,0,0,1,0,0,0,0;
               1' 0,0,0,0,1,0,0,0;
               2' 0,0,0,0,0,1,0,0;
               3' 0,0,0,0,0,0,1,0;
               4' 0,0,0,0,0,0,0,1]
            '''
            # 将one hot向量移到GPU上
            self.one_hot_feat = one_hot_feat.cuda() if self.has_cuda else one_hot_feat
            # 生成one-hot问题向量，大小[概念数，概念数]
            self.one_hot_q = torch.eye(self.concept_num, device=self.one_hot_feat.device)
            '''
                  1,2,3,4
              1  [1,0,0,0;
              2   0,1,0,0;
              3   0,0,1,0;
              4   0,0,0,1]
            '''

            # size:1 大小为1行，concept_num列全0 [0,0,0,0]
            zero_padding = torch.zeros(1, self.concept_num, device=self.one_hot_feat.device)
            # 通过上面可以看得出两个one-hot大小并不一致，所以利用zero_padding对问题one_hot_q进行拼
            self.one_hot_q = torch.cat((self.one_hot_q, zero_padding), dim=0)
            '''
                 1,2,3,4
                 1  [1,0,0,0;
                 2   0,1,0,0;
                 3   0,0,1,0;
                 4   0,0,0,1;
                 zp  0,0,0,0]   
            '''
            # concept and concept & response embeddings
            self.emb_x = nn.Embedding(self.res_len * concept_num, embedding_dim)
            # batch_size=1，seq_length=res_len * concept_num（假设8）;将输入反应回答次数*概念数，进行嵌入低维度向量表示，dim为输入参数：嵌入维度32
            '''
            输入的值范围是[0,7]
            若输入向量是[2，4]，输入的嵌入维度是32
            则生成的嵌入矩阵是[2,4,32]
            即，两维张量，每一维是4行，32列
            '''
            # last embedding is used for padding, so dim + 1
            # embedding处理的是张量，（batch_size,seq_length）的整数类型输入张量，所以在这里，seq_length=concept_num+1;batch_size=1
            self.emb_c = nn.Embedding(concept_num + 1, embedding_dim, padding_idx=-1)
            # 一次性对整个序列进行处理，将序列中的每个整数值映射到一个低维浮点数向量中，每一行是一个嵌入向量
            # f_self function and f_neighbor functions
            # 他们是用于计算节点的嵌入向量的函数，这些向量将用于节点分类任务
            mlp_input_dim = hidden_dim + embedding_dim
            # mlp的输入维度是隐藏层维度加嵌入层维度 是64（32+32）
            self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
            # 表示定义一个多层感知机（MLP）用于计算节点自身的输出。输入维度64，输出维度维hidden_dim=32；他将接收节点的特征向量和上一步隐藏状态作为输入，并输出更新后的隐藏状态
            self.f_neighbor_list = nn.ModuleList()
            # 一个空的ModuleList（），用于存储f_neighbor函数的列表，每一个模块/函数用于计算不同类型的边的嵌入向量
            if graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:  # 先判断图类型
                # f_in and f_out functions 使用两个MLP函数来计算邻居节点的嵌入向量
                self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
                # 表示将f_in函数加入到这个函数列表中，用于计算邻居节点的嵌入向量
                # 输入张量大小[2*(mlpdim)=128],输出[32]
                self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
                # 表示将f_out函数加入到这个函数列表中，用于计算并输出
            else:  # ['MHA', 'VAE']
                for i in range(edge_type_num):
                    # 使用多个MLP函数，循环迭代所有边的类型，进行边的邻居节点的嵌入向量计算
                    # 循环到每一条边，产生一个f_neighbor的函数加入到函数列表中，并计算出当前边类型的邻居节点的嵌入向量
                    # 输入张量[128],输出张量[32]
                    self.f_neighbor_list.append(
                        MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

            # Erase & Add Gate 作为gkt中的擦除添加门，输入维度为32（hidden_dim）,输出维度为concept_num
            self.erase_add_gate = EraseAddGate(hidden_dim, concept_num)
            '''
            erase_add_gate = EraseAddGate(hidden_dim, concept_num)
            hidden_state = torch.randn(batch_size, hidden_dim)
            prev_knowledge_state = torch.randn(batch_size, concept_num)
            new_knowledge_state = erase_add_gate(hidden_state, prev_knowledge_state)
            自定义的擦除添加门，用来控制模型对先前学生知识状态进行擦除/添加的过程，将输入的隐藏状态和先前学生知识状态作为输入，通过两个全连接层（分别用于擦除和添加）输出一个向量，更新学生的知识状态
            '''
            # Gate Recurrent Unit 作为gkt的门控循环单元
            self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
            '''
            自带门控循环单元模块，他接受一个输入张量和上一个时间步的隐藏状态，输出当前时间步的隐藏状态，在gkt中，这个模块被用来更新学生的隐藏状态
            gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
            input_tensor = torch.randn(batch_size, input_dim)
            prev_hidden_state = torch.randn(batch_size, hidden_dim)
            hidden_state = gru(input_tensor, prev_hidden_state)
            '''
            # prediction layer 全连接层，将隐藏层的维度32（通过一个线性变换）转换为一个标量，用于预测学生是否掌握该知识点，输出维度为1
            # predict = nn.Linear(hidden_dim, 1, bias=bias)
            # hidden_state = torch.randn(batch_size, hidden_dim)
            # prediction = predict(hidden_state)
            self.predict = nn.Linear(hidden_dim, 1, bias=bias)

        # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
        def _agg_neighbors(self, tmp_ht, qt):
            r"""
            Parameters:
                tmp_ht: temporal hidden representations of all concepts after the aggregate step
                qt: question indices for all students in a batch at the current timestamp
            Shape:
                tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
                qt: [batch_size]
                m_next: [batch_size, concept_num, hidden_dim]
            Return:
                m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
                concept_embedding: input of VAE (optional)
                rec_embedding: reconstructed input of VAE (optional)
                z_prob: probability distribution of latent variable z in VAE (optional)
            """
            qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
            masked_qt = qt[qt_mask]  # [mask_num, ]
            masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, concept_num, hidden_dim + embedding_dim]
            mask_num = masked_tmp_ht.shape[0]
            self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
            self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + embedding_dim]
            self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]
            expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.concept_num, 1)  #[mask_num, concept_num, hidden_dim + embedding_dim]
            neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)  #[mask_num, concept_num, 2 * (hidden_dim + embedding_dim)]
            concept_embedding, rec_embedding, z_prob = None, None, None

            if self.graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:
                adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
                reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
                # self.f_neighbor_list[0](neigh_ht) shape: [mask_num, concept_num, hidden_dim]
                neigh_features = adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * self.f_neighbor_list[1](neigh_ht)
            else:  # ['MHA', 'VAE']
                concept_index = torch.arange(self.concept_num, device=qt.device)
                concept_embedding = self.emb_c(concept_index)  # [concept_num, embedding_dim]
                if self.graph_type == 'MHA':
                    query = self.emb_c(masked_qt)
                    key = concept_embedding
                    att_mask = Variable(torch.ones(self.edge_type_num, mask_num, self.concept_num, device=qt.device))
                    for k in range(self.edge_type_num):
                        index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
                        att_mask[k] = att_mask[k].index_put(index_tuple, torch.zeros(mask_num, device=qt.device))
                    graphs = self.graph_model(masked_qt, query, key, att_mask)
                else:  # self.graph_type == 'VAE'
                    sp_send, sp_rec, sp_send_t, sp_rec_t = self._get_edges(masked_qt)
                    graphs, rec_embedding, z_prob = self.graph_model(concept_embedding, sp_send, sp_rec, sp_send_t, sp_rec_t)
                neigh_features = 0
                for k in range(self.edge_type_num):
                    adj = graphs[k][masked_qt, :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
                    if k == 0:
                        neigh_features = adj * self.f_neighbor_list[k](neigh_ht)
                    else:
                        neigh_features = neigh_features + adj * self.f_neighbor_list[k](neigh_ht)
                if self.graph_type == 'MHA':
                    neigh_features = 1. / self.edge_type_num * neigh_features
            # neigh_features: [mask_num, concept_num, hidden_dim]
            m_next = tmp_ht[:, :, :self.hidden_dim]
            m_next[qt_mask] = neigh_features
            m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
            return m_next, concept_embedding, rec_embedding, z_prob

        # Update step, as shown in Section 3.3.2 of the paper
        def _update(self, tmp_ht, ht, qt):
            r"""
            Parameters:
                tmp_ht: temporal hidden representations of all concepts after the aggregate step
                ht: hidden representations of all concepts at the current timestamp
                qt: question indices for all students in a batch at the current timestamp
            Shape:
                tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
                ht: [batch_size, concept_num, hidden_dim]
                qt: [batch_size]
                h_next: [batch_size, concept_num, hidden_dim]
            Return:
                h_next: hidden representations of all concepts at the next timestamp
                concept_embedding: input of VAE (optional)
                rec_embedding: reconstructed input of VAE (optional)
                z_prob: probability distribution of latent variable z in VAE (optional)
            """
            qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
            mask_num = qt_mask.nonzero().shape[0]
            # GNN Aggregation
            m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht, qt)  # [batch_size, concept_num, hidden_dim]
            # Erase & Add Gate
            m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, concept_num, hidden_dim]
            # GRU
            h_next = m_next
            res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim), ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * concept_num, hidden_num]
            index_tuple = (torch.arange(mask_num, device=qt_mask.device), )
            h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.concept_num, self.hidden_dim))
            return h_next, concept_embedding, rec_embedding, z_prob

        # Predict step, as shown in Section 3.3.3 of the paper
        # 预测下一个时间步的答题情况？
        # 用在更新概念状态后预测每一个学生在下一个时间戳中各个知识点的掌握情况
        def _predict(self, h_next, qt):
            r"""
            Parameters:
                h_next: hidden representations of all concepts at the next timestamp after the update step
                qt: question indices for all students in a batch at the current timestamp
            Shape:
                h_next: [batch_size, concept_num, hidden_dim]
                qt: [batch_size]
                y: [batch_size, concept_num]
            Return:
                y: predicted correct probability of all concepts at the next timestamp
            """
            qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
            y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, concept_num]
            y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
            return y

        # predicate方法最后返回的是对下一时间戳所有概念掌握情况的预测，即掌握，没掌握；
        # 再用知识点的掌握情况去预测学生下一个时间戳问题的回答情况，所以需要下一个时间戳的问题索引，且保证当前时间下，学生回答了问题。
        def _get_next_pred(self, yt, q_next):
            r"""
            Parameters:
                yt: predicted correct probability of all concepts at the next timestamp
                q_next: question index matrix at the next timestamp
                batch_size: the size of a student batch
            Shape:
                y: [batch_size, concept_num]
                questions: [batch_size, seq_len]
                pred: [batch_size, ]
            Return:
                pred: predicted correct probability of the question answered at the next timestamp
            """
            next_qt = q_next
            next_qt = torch.where(next_qt != -1, next_qt, self.concept_num * torch.ones_like(next_qt, device=yt.device))
            one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)  # [batch_size, concept_num]
            # dot product between yt and one_hot_qt
            pred = (yt * one_hot_qt).sum(dim=1)  # [batch_size, ]
            return pred

        # Get edges for edge inference in VAE
        def _get_edges(self, masked_qt):
            r"""
            Parameters:
                masked_qt: qt index with -1 padding values removed
            Shape:
                masked_qt: [mask_num, ]
                rel_send: [edge_num, concept_num]
                rel_rec: [edge_num, concept_num]
            Return:
                rel_send: from nodes in edges which send messages to other nodes
                rel_rec:  to nodes in edges which receive messages from other nodes
            """
            mask_num = masked_qt.shape[0]
            row_arr = masked_qt.cpu().numpy().reshape(-1, 1)  # [mask_num, 1]
            row_arr = np.repeat(row_arr, self.concept_num, axis=1)  # [mask_num, concept_num]
            col_arr = np.arange(self.concept_num).reshape(1, -1)  # [1, concept_num]
            col_arr = np.repeat(col_arr, mask_num, axis=0)  # [mask_num, concept_num]
            # add reversed edges
            new_row = np.vstack((row_arr, col_arr))  # [2 * mask_num, concept_num]
            new_col = np.vstack((col_arr, row_arr))  # [2 * mask_num, concept_num]
            row_arr = new_row.flatten()  # [2 * mask_num * concept_num, ]
            col_arr = new_col.flatten()  # [2 * mask_num * concept_num, ]
            data_arr = np.ones(2 * mask_num * self.concept_num)
            init_graph = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(self.concept_num, self.concept_num))
            init_graph.setdiag(0)  # remove self-loop edges
            row_arr, col_arr, _ = sp.find(init_graph)
            row_tensor = torch.from_numpy(row_arr).long()
            col_tensor = torch.from_numpy(col_arr).long()
            one_hot_table = torch.eye(self.concept_num, self.concept_num)
            rel_send = F.embedding(row_tensor, one_hot_table)  # [edge_num, concept_num]
            rel_rec = F.embedding(col_tensor, one_hot_table)  # [edge_num, concept_num]
            sp_rec, sp_send = rel_rec.to_sparse(), rel_send.to_sparse()
            sp_rec_t, sp_send_t = rel_rec.T.to_sparse(), rel_send.T.to_sparse()
            sp_send = sp_send.to(device=masked_qt.device)
            sp_rec = sp_rec.to(device=masked_qt.device)
            sp_send_t = sp_send_t.to(device=masked_qt.device)
            sp_rec_t = sp_rec_t.to(device=masked_qt.device)
            return sp_send, sp_rec, sp_send_t, sp_rec_t

        def forward(self, features, questions):
            r"""
            Parameters:
                features: input one-hot matrix
                questions: question index matrix
            seq_len dimension needs padding, because different students may have learning sequences with different lengths.
            Shape:
                features: [batch_size, seq_len]
                questions: [batch_size, seq_len]
                pred_res: [batch_size, seq_len - 1]
            Return:
                pred_res: the correct probability of questions answered at the next timestamp
                concept_embedding: input of VAE (optional)
                rec_embedding: reconstructed input of VAE (optional)
                z_prob: probability distribution of latent variable z in VAE (optional)
            """
            batch_size, seq_len = features.shape
            ht = Variable(torch.zeros((batch_size, self.concept_num, self.hidden_dim), device=features.device))
            pred_list = []
            ec_list = []  # concept embedding list in VAE
            rec_list = []  # reconstructed embedding list in VAE
            z_prob_list = []  # probability distribution of latent variable z in VAE
            for i in range(seq_len):
                xt = features[:, i]  # [batch_size]
                qt = questions[:, i]  # [batch_size]
                qt_mask = torch.ne(qt, -1)  # [batch_size], next_qt != -1
                tmp_ht = self._aggregate(xt, qt, ht, batch_size)  # [batch_size, concept_num, hidden_dim + embedding_dim]
                h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht, qt)  # [batch_size, concept_num, hidden_dim]
                ht[qt_mask] = h_next[qt_mask]  # update new ht
                yt = self._predict(h_next, qt)  # [batch_size, concept_num]
                if i < seq_len - 1:
                    pred = self._get_next_pred(yt, questions[:, i + 1])
                    pred_list.append(pred)
                ec_list.append(concept_embedding)
                rec_list.append(rec_embedding)
                z_prob_list.append(z_prob)
            pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]
            return pred_res, ec_list, rec_list, z_prob_list


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    NOTE: Stole and modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
    """

    def __init__(self, n_head, concept_num, input_dim, d_k, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.concept_num = concept_num
        self.d_k = d_k
        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        # inferred latent graph, used for saving and visualization
        self.graphs = nn.Parameter(torch.zeros(n_head, concept_num, concept_num))
        self.graphs.requires_grad = False

    def _get_graph(self, attn_score, qt):
        r"""
        Parameters:
            attn_score: attention score of all queries
            qt: masked question index
        Shape:
            attn_score: [n_head, mask_num, concept_num]
            qt: [mask_num]
        Return:
            graphs: n_head types of inferred graphs
        """
        graphs = Variable(torch.zeros(self.n_head, self.concept_num, self.concept_num, device=qt.device))
        for k in range(self.n_head):
            index_tuple = (qt.long(), )
            graphs[k] = graphs[k].index_put(index_tuple, attn_score[k])  # used for calculation
            #############################
            # here, we need to detach edges when storing it into self.graphs in case memory leak!
            self.graphs.data[k] = self.graphs.data[k].index_put(index_tuple, attn_score[k].detach())  # used for saving and visualization
            #############################
        return graphs

    def forward(self, qt, query, key, mask=None):
        r"""
        Parameters:
            qt: masked question index
            query: answered concept embedding for a student batch
            key: concept embedding matrix
            mask: mask matrix
        Shape:
            qt: [mask_num]
            query: [mask_num, embedding_dim]
            key: [concept_num, embedding_dim]
        Return:
            graphs: n_head types of inferred graphs
        """
        d_k, n_head = self.d_k, self.n_head
        len_q, len_k = query.size(0), key.size(0)

        # Pass through the pre-attention projection: lq x (n_head *dk)
        # Separate different heads: lq x n_head x dk
        q = self.w_qs(query).view(len_q, n_head, d_k)
        k = self.w_ks(key).view(len_k, n_head, d_k)

        # Transpose for attention dot product: n_head x lq x dk
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        attn_score = self.attention(q, k, mask=mask)  # [n_head, mask_num, concept_num]
        graphs = self._get_graph(attn_score, qt)
        return graphs


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, msg_hidden_dim, msg_output_dim, concept_num, edge_type_num=2,
                 tau=0.1, factor=True, dropout=0., bias=True):
        super(VAE, self).__init__()
        self.edge_type_num = edge_type_num
        self.concept_num = concept_num
        self.tau = tau
        self.encoder = MLPEncoder(input_dim, hidden_dim, output_dim, factor=factor, dropout=dropout, bias=bias)
        self.decoder = MLPDecoder(input_dim, msg_hidden_dim, msg_output_dim, hidden_dim, edge_type_num, dropout=dropout, bias=bias)
        # inferred latent graph, used for saving and visualization
        self.graphs = nn.Parameter(torch.zeros(edge_type_num, concept_num, concept_num))
        self.graphs.requires_grad = False

    def _get_graph(self, edges, sp_rec, sp_send):
        r"""
        Parameters:
            edges: sampled latent graph edge weights from the probability distribution of the latent variable z
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send: one-hot encoded send-node index(sparse tensor)
        Shape:
            edges: [edge_num, edge_type_num]
            sp_rec: [edge_num, concept_num]
            sp_send: [edge_num, concept_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
        """
        x_index = sp_send._indices()[1].long()  # send node index: [edge_num, ]
        y_index = sp_rec._indices()[1].long()   # receive node index [edge_num, ]
        graphs = Variable(torch.zeros(self.edge_type_num, self.concept_num, self.concept_num, device=edges.device))
        for k in range(self.edge_type_num):
            index_tuple = (x_index, y_index)
            graphs[k] = graphs[k].index_put(index_tuple, edges[:, k])  # used for calculation
            #############################
            # here, we need to detach edges when storing it into self.graphs in case memory leak!
            self.graphs.data[k] = self.graphs.data[k].index_put(index_tuple, edges[:, k].detach())  # used for saving and visualization
            #############################
        return graphs

    def forward(self, data, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            data: input concept embedding matrix
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
        Shape:
            data: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]
            sp_rec: [edge_num, concept_num]
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
            output: the reconstructed data
            prob: q(z|x) distribution
        """
        logits = self.encoder(data, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [edge_num, output_dim(edge_type_num)]
        edges = gumbel_softmax(logits, tau=self.tau, dim=-1)  # [edge_num, edge_type_num]
        prob = F.softmax(logits, dim=-1)
        output = self.decoder(data, edges, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [concept_num, embedding_dim]
        graphs = self._get_graph(edges, sp_send, sp_rec)
        return graphs, output, prob
