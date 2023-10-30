from torch.utils.data.dataset import Dataset
from typing import Any
from datetime import datetime
import drms


class SHARPDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def __len__(self):
        return 0
    

class SolarFlaresData:
    def __init__(self):
        self.client = drms.Client()

    def get_data(self, series_name:str, active_region_number:str, condition:str, date_time:datetime, keys:list, segment:str):
        keys, segments = self.client.query('hmi.sharp_cea_720s[86][2010.07.14_11:12:00/12h][? (QUALITY<65536) ?]', key='T_REC, USFLUXL, ERRVF', seg='Br')
        print (keys, segments)

class SMARPDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def __len__(self):
        return 0


if __name__ == "__main__":
    sf = SolarFlaresData()
