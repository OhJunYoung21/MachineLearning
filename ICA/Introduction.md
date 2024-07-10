## ICA

### Issue 정리

1️⃣ fMRIprep 의 결과물로 생기는 .tsv파일 읽어오기

~~~python3
import pandas as pd
pd.read_csv('path_to_your_file',sep='\t')
~~~

sep='\t'를 하지 않으면 파일을 확인하기 어려워진다. sep = '\t'는 seperate = tab. 즉, 탭으로 구분된 데이터를 읽어오겠다는 의미이다.