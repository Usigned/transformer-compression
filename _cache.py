class StrategyCache:
    def __init__(self, s_len, fpath=None, flush_on_write=True) -> None:
        self.data:dict[tuple, float] = {}
        self.s_len = s_len
        self.fpath = fpath
        self.flush_on_write = flush_on_write
        self.initialize()

    def initialize(self):
        if self.fpath:
            self.load_from_csv()

    def __item__(self, key):
        return self.data[key]
    
    def _load_from_csv(self, fpath):
        data = {}
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.split(',')
                s, acc = line[:-1], line[-1]
                s = tuple(int(e.strip()) for e in s)
                assert len(s) == self.s_len, f'{len(s)} != {self.s_len}, s: {s}'
                acc = float(acc.strip())
                data[s] = acc
        self.data.update(data)

    def _save_to_csv(self, fpath, overwrite=True):
        mod = 'w' if overwrite else 'a+'
        with open(fpath, mod) as f:
            for s, acc in self.data.items():
                line = ','.join((str(num) for num in (*s, acc)))
                f.write(line+'\n')
                f.flush()

    def load_from_csv(self):
        if not self.fpath:
            raise RuntimeError('fpath not set')
        self._load_from_csv(self.fpath)
    
    def save_to_csv(self):
        if not self.fpath:
            raise RuntimeError('fpath not set')
        self._save_to_csv(self.fpath)
    
    def __contains__(self, key):
        return key in self.data.keys()
    
    def get(self, key):
        return self[key]

    def __getitem__(self, key):
        key = tuple(key)
        assert len(key) == self.s_len, f'len {self.s_len} expected, got {len(key)}'
        return self.data[key]

    def put(self, strategy, acc):
        self[strategy] = acc

    def __setitem__(self, strategy, acc):
        strategy = tuple(strategy)
        assert len(strategy) == self.s_len
        self.data[strategy] = float(acc)
        if self.flush_on_write:
            self.flush()

    def flush(self):
        if self.fpath:
            self.save_to_csv()

if __name__ == '__main__':
    cache = StrategyCache(60, 'strategy.csv')
    s = [2,2,2,4,2,2,2,2,2,2,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
    cache[s] = 0.870199978351593
