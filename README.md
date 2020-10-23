# Banker BERT

Banka şikayet metni sınıflandırma için fine-tune'lanmış BERT. 

Çalıştırmak için lütfen [buradan](https://drive.google.com/file/d/1y_Ym4tij9C_esgbW8XzzA6FrcPYNd_Mh/view?usp=sharing) state.pt dosyasını indirip gördüğünüz bu dosyalar ile aynı directory'e koyun ve app.py'yi bir yerel sunucuda host'layın. Örneğin uvicorn kullanıyorsanız dosyaların bulunduğu directory'de cmd'ye 
```
uvicorn main:app --reload
```
yazmanız yeterli olacaktır. Sonra ise http://127.0.0.1:8000/docs adresinden API'yı deneyebilirsiniz. Bunun için Python 3.6.7'ye ve requirements.txt'deki kütüphanelere sahip olmanız gerekmektedir.

Bunlarla uğraşamam diyorsanız Docker image'ına [buradan](https://hub.docker.com/repository/docker/demegire/sumooo) erişebilirsiniz. Bu image'ı indirip kendi Docker'ınızda çalıştırıp localhost/docs adresinde API'ya erişebilirsiniz. 

Detaylı bilgi için lütfen [raporu](https://colab.research.google.com/drive/1J-q5sP8CQHIBlJan21gOrKw9NihbGS85?usp=sharing) okuyun.
