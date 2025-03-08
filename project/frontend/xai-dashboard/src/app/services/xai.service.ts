// src/app/xai.service.ts

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class XaiService {
  private apiUrl = 'http://localhost:8000/xai';

  constructor(private http: HttpClient) { }

  getIntegratedGradients(sample: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/integrated_gradients`, { data: sample });
  }

  getLimeExplanation(sample: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/lime`, { data: sample });
  }

  getShapExplanation(sample: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/shap`, { data: sample });
  }

  getDecisionTreeExplanation(sample: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/decision_tree`, { data: sample });
  }
}
