import unittest
from HW2_2018384 import minFunc

class testpoint(unittest.TestCase):
	def test_minFunc(self):
		self.assertEqual(minFunc(2,'(2,3,1) d -'),'W+X')
		self.assertEqual(minFunc(2,'(2,1) d(3)'),'W+X')
		self.assertEqual(minFunc(3,'(2,3,4,7) d(6)'),'X+WY`')
		self.assertEqual(minFunc(4,'(0,1,3,5,9,12) d(2,4,6,7)'),'W`+X`Y`Z+XY`Z`')
		self.assertEqual(minFunc(4,'(0,2,5,8,7,10,13,15) d -'),'X`Z`+XZ')
                
if __name__=='__main__':
	unittest.main()
