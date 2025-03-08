import { TestBed } from '@angular/core/testing';

import { XaiService } from './xai.service';

describe('XaiService', () => {
  let service: XaiService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(XaiService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
