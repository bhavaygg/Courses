#!/usr/bin/env python3
# CSE 101 - IP HW3
# K-Map Minimization 
# Name: Bhavay Aggarwal
# Roll Number: 2018384
# Section: B
# Group: 1
import re
import itertools

ROLLNUM_REGEX = "201[0-9]{4}"

class Graph(object):
    name = "Bhavay"
    email = "bhavay18384@iiitd.ac.in"
    roll_num = "2018384"
    list1=[]
    lists=[]
    listf=[]
    n=0
    end=0
    start=0
    dict={}

    def __init__ (self, vertices, edges):
        """
        Initializes object for the class Graph

        Args:
            vertices: List of integers specifying vertices in graph
            edges: List of 2-tuples specifying edges in graph
        """
        global dict
        global n
        self.vertices = vertices
        
        ordered_edges = list(map(lambda x: (min(x), max(x)), edges))
        self.n=len(vertices)
        self.edges    = ordered_edges
        for i in vertices:
            self.dict[i]=[]
        for i in edges:
            self.dict[i[0]].append(i[1])
            self.dict[i[1]].append(i[0])    
        self.validate()

    def validate(self):
        """
        Validates if Graph if valid or not

        Raises:
            Exception if:
                - Name is empty or not a string
                - Email is empty or not a string
                - Roll Number is not in correct format
                - vertices contains duplicates
                - edges contain duplicates
                - any endpoint of an edge is not in vertices
        """

        if (not isinstance(self.name, str)) or self.name == "":
            raise Exception("Name can't be empty")

        if (not isinstance(self.email, str)) or self.email == "":
            raise Exception("Email can't be empty")

        if (not isinstance(self.roll_num, str)) or (not re.match(ROLLNUM_REGEX, self.roll_num)):
            raise Exception("Invalid roll number, roll number must be a string of form 201XXXX. Provided roll number: {}".format(self.roll_num))

        if not all([isinstance(node, int) for node in self.vertices]):
            raise Exception("All vertices should be integers")

        elif len(self.vertices) != len(set(self.vertices)):
            duplicate_vertices = set([node for node in self.vertices if self.vertices.count(node) > 1])

            raise Exception("Vertices contain duplicates.\nVertices: {}\nDuplicate vertices: {}".format(vertices, duplicate_vertices))

        edge_vertices = list(set(itertools.chain(*self.edges)))

        if not all([node in self.vertices for node in edge_vertices]):
            raise Exception("All endpoints of edges must belong in vertices")

        if len(self.edges) != len(set(self.edges)):
            duplicate_edges = set([edge for edge in self.edges if self.edges.count(edge) > 1])

            raise Exception("Edges contain duplicates.\nEdges: {}\nDuplicate vertices: {}".format(edges, duplicate_edges))

    def min_dist(self, start_node, end_node):
        '''
        Finds minimum distance between start_node and end_node

        Args:
            start_node: Vertex to find distance from
            end_node: Vertex to find distance to

        Returns:
            An integer denoting minimum distance between start_node
            and end_node
        '''

        raise NotImplementedError

    def all_shortest_paths(self):
        """
        Finds all shortest paths between start_node and end_node

        Args:
            start_node: Starting node for paths
            end_node: Destination node for paths

        Returns:
            A list of path, where each path is a list of integers.
        """
        global lists
        min=10000
        l=[]
        for i in self.lists:
           if len(i)<min:
                min = len(i)
        for i in self.lists:
            if len(i)==min:
                l.append(i)     
        self.lists=l 

    def all_paths(self, var):
        """
        Finds all paths from node to destination with length = dist

        Args:
            node: Node to find path from
            destination: Node to reach
            dist: Allowed distance of path
            path: path already traversed

        Returns:
            List of path, where each path is list ending on destination

            Returns None if there no paths
        """
        global list1
        global listf
        global lists
        self.var=var
        for k in self.dict[var]:
            sum=0
            if k==self.end:
                self.list1.append(var)
                for i in self.list1:
                    if i == var:
                        self.listf.append(i)
                        break
                    else:
                        self.listf.append(i)
                if not self.listf==False:
                    if self.listf[0]!=self.start:
                        self.listf = [self.start]+self.listf   
                if self.listf.count(self.start)==1:           
                    self.lists.append(self.listf) 
                self.listf=[]
            else:
                if k not in self.list1:
                    if var not in self.list1:
                        self.list1.append(var)   
                    var1=k
                    for m in self.dict[var1]:
                        for n in self.list1:
                            if m==n:
                                sum+=1              
                    if sum!=len(self.dict[var1]):        
                        self.all_paths(var1)
                        if len(self.list1)>0:
                            x = self.list1.index(var)
                            self.list1=self.list1[:x]

    def betweenness_centrality(self,node):
        """
        Find betweenness centrality of the given node

        Args:
            node: Node to find betweenness centrality of.

        Returns:
            Single floating point number, denoting betweenness centrality
            of the given node
        """
        global lists
        count=0
        for i in self.lists:
            for j in i:
                if j == node:
                    count+=1
        central=count/len(self.lists)
        return central/((self.n-1)*(self.n-2)/2) 


    def top_k_betweenness_centrality(self):
        """
        Find top k nodes based on highest equal betweenness centrality.

        
        Returns:
            List a integer, denoting top k nodes based on betweenness
            centrality.
        """
        global lists
        global list1
        sum1=0
        for k in vertices:
            for i in range (0,len(vertices)):
                if k!=vertices[i]:  
                    for j in range (i+1,len(vertices)):
                        if k!=vertices[j]:
                            self.start=vertices[i]
                            self.end=vertices[j]
                            start1=vertices[i]
                            node=k
                            self.all_paths(start1)
                            self.all_shortest_paths()
                            sum1+=self.betweenness_centrality(node)
                        self.lists=[]
                        self.list1=[]
            dict1[k]=round(sum1,5)                   
            sum1=0
        v=list(dict1.values())
        k=list(dict1.keys())
        y=max(v)
        for i in dict1:
            if dict1[i]==y:
                print(i,":",dict1[i])

dict1={}
if __name__ == "__main__":
    vertices = [1, 2, 3, 4, 5, 6]
    edges    = [(1, 2), (1, 5), (2, 3), (2, 5), (3, 4), (3, 6), (4, 5), (4, 6)]
    graph = Graph(vertices, edges)
    graph.top_k_betweenness_centrality()
    
