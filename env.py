# -*- coding:utf8 -*-
import os
import cv2
import numpy as np


class AutoNavEnv(object):
    def __init__(self):
        self._pos = [0, 0]
        self._dir = 0  # 0 front 1 right 2 back 3 left

        self.__ops = 0
        self.__mvs = 0
        self.__colli = 0

        self.env_name = ''
        self.room_map = dict()

    def __move(self, delta):
        npos = [self._pos[0], self._pos[1]]
        if self._dir == 0:
            npos[0] += delta
        elif self._dir == 1:
            npos[1] += delta
        elif self._dir == 2:
            npos[0] -= delta
        else:
            npos[1] -= delta

        if (npos[0], npos[1]) in self.room_map:
            self._pos = npos
            return True
        else:
            return False

    def dir2name(self, direction=None):
        if direction is None:
            direction = self._dir
        if direction == 0:
            return 'FRONT'
        elif direction == 1:
            return 'RIGHT'
        elif direction == 2:
            return 'BACK'
        else:
            return 'LEFT'

    @property
    def observation(self):
        """
        :return: (image_left, image_right)
        """
        dir_name = self.dir2name()
        left = "{}_{}_{}_{}.png".format(self._pos[0], self._pos[1],
                                        dir_name, 'LEFT')
        right = "{}_{}_{}_{}.png".format(self._pos[0], self._pos[1],
                                         dir_name, 'RIGHT')

        p_left = os.path.join(self.env_name, left)
        p_right = os.path.join(self.env_name, right)

        img_l = cv2.imread(p_left, cv2.IMREAD_COLOR)
        img_r = cv2.imread(p_right, cv2.IMREAD_COLOR)
        img_l = np.float32(img_l)
        img_r = np.float32(img_r)
        return img_l, img_r

    @property
    def position(self):
        """
        :return: discrete position x, y
        """
        return self._pos[0], self._pos[1]

    @property
    def direction(self):
        """
        :return: 0 front 1 right 2 back 3 left
        """
        return self._dir

    @property
    def num_of_moves(self):
        return self.__mvs

    @property
    def num_of_operations(self):
        return self.__ops

    @property
    def num_of_collisions(self):
        return self.__colli

    def make(self, env_name: 'environment folder'):
        self.env_name = env_name
        for _, __, names in os.walk(env_name):
            assert len(names) != 0
            for name in names:
                if name[-3:] != 'png':
                    continue
                x, y, _, _ = name[:-4].split('_')
                x, y = int(x), int(y)
                if (x, y) not in self.room_map:
                    self.room_map[x, y] = 1
        return self

    def reset(self, st_pos=(1, 1), st_dir=0):
        self._pos = list(st_pos)
        self._dir = st_dir
        if (st_pos[0], st_pos[1]) not in self.room_map:
            print("Invalid Start Position, Restart with default")
            for k in self.room_map:
                self._pos = k
                break

        self.__ops = 0
        self.__mvs = 0
        self.__colli = 0
        return self.observation

    def render(self):
        imgl, imgr = self.observation
        img = np.concatenate([imgl.T, imgr.T], axis=1)
        img = img.T
        s = img.shape
        img = cv2.resize(img, (int(s[1]/2), int(s[0]/2)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("AutoNav", np.uint8(img))
        cv2.waitKey(1)

    def step(self, action):
        """
        :param action: 0 forward 1 backward 2 turn left 3 turn right
        :return: observation [1 walkable, -1 walkable], reward, done, info
        """
        self.__ops += 1

        success_mv = True
        if action == 0:
            success_mv = self.__move(1)
            self.__mvs += 1
        elif action == 1:
            success_mv = self.__move(-1)
            self.__mvs += 1
        elif action == 2:
            self._dir = (self._dir + 1) % 4
        elif action == 3:
            self._dir = (self._dir + 3) % 4
        else:
            raise Exception("Action Out of Bound")

        if success_mv:
            info = 'successful'
            rew = 1
        else:
            info = 'blocked'
            rew = -1
            self.__colli += 1

        return self.observation, rew, False, info


if __name__ == "__main__":
    env = AutoNavEnv()
    env.make('images')
    observation = env.reset()
    #
    import random
    import time
    for i in range(1000):
        a = random.randint(0, 3)
        env.step(a)
        print(i, 'step', a, env._pos, env._dir)
        env.render()

    cv2.destroyAllWindows()



